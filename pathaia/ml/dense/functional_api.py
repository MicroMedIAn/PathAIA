# coding: utf8
"""
A module to learn and predict dense descriptors.

Can be used to create ml-based filters for image analysis and patch extraction.
"""
from ..util import images_in_folder, dataset2folders
from ...patches.util import unlabeled_regular_grid_list
from sklearn.cluster import MiniBatchKMeans, KMeans
import numpy
from numpy.random import shuffle
from matplotlib import pyplot as plt
import os
import pickle
import csv
import shutil
from skimage.io import imread, imsave


def sample_img(image, psize, spl_per_image):
    """
    Fit vocabulary on a single image.

    *********************************
    """
    img = image.astype(float)
    spaceshape = (image.shape[0], image.shape[1])
    positions = unlabeled_regular_grid_list(spaceshape, psize)
    shuffle(positions)
    positions = positions[0:spl_per_image]
    patches = [img[i:i + psize, j:j + psize].reshape(-1) for i, j in positions]
    return patches


def fit_on_image(image, vocabulary, psize, spl_per_image):
    """
    Fit vocabulary on a single image.

    *********************************
    """
    patches = sample_img(image, psize, spl_per_image)
    vocabulary.partial_fit(patches)


def fit_on_imbatch(images, vocabulary, psize, spl_per_image):
    """
    Fit vocabulary on a batch of images.

    ************************************
    """
    patches = []
    for img in images:
        patches += sample_img(img, psize, spl_per_image)
    vocabulary.partial_fit(patches)


def fit_on_slide(ptc_folder, vocabulary, voclen, psize, spl_per_image):
    """
    Create a texture vocabulary.

    Arguments:
        - ptc_folder: str, path to an image folder.
        - psize: int, size of window in pixels.
        - dictlen: int, number of words in vocabulary.
        - verbose: int, verbosity.

    """
    imgs = []
    for image in images_in_folder(ptc_folder, randomize=True, datalim=voclen):
        imgs.append(image)
    fit_on_imbatch(imgs, vocabulary, psize, spl_per_image)


def learn_vocabulary(projfolder, outfolder, level,
                     psize=8, voclen=256, spl_per_image=100,
                     slide_data_lim=100, verbose=2, reassignment_ratio=0.):
    """
    Fit vocabulary on a single entire slide dataset.

    *********************************
    """
    vocabulary = MiniBatchKMeans(n_clusters=voclen, reassignment_ratio=reassignment_ratio)
    slide2folder = dataset2folders(projfolder, level, randomize=True,
                                   slide_data_lim=slide_data_lim)
    k = 0
    for slidename, folder in slide2folder.items():
        k += 1
        if verbose > 0:
            print("processing slide {} / {}".format(k, len(slide2folder)))
            print("fitting on slide name: {}".format(slidename))
            print("fitting on level: {}".format(level))
        fit_on_slide(folder, vocabulary, voclen, psize, spl_per_image)

    # affichage
    fig = plt.figure()
    patch_shape = (psize, psize, 3)
    for i in range(int(numpy.sqrt(voclen))):
        for j in range(int(numpy.sqrt(voclen))):
            image = vocabulary.cluster_centers_[i * int(numpy.sqrt(voclen)) + j].reshape(patch_shape)
            print("stats of filter {}:\nmin={}, max={}, mean={}, dynamic={}".format(i * int(numpy.sqrt(voclen)) + j,
                                                                                    image.min(),
                                                                                    image.max(),
                                                                                    image.mean(),
                                                                                    len(numpy.unique(image))))
            ax = fig.add_subplot(int(numpy.sqrt(voclen)), int(numpy.sqrt(voclen)), i * int(numpy.sqrt(voclen)) + j + 1)
            # Turn off tick labels
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            image = numpy.around(image)
            image = image.astype(numpy.uint8)
            ax.imshow(image)

    plt.savefig(os.path.join(outfolder, "vocabulary_{}_{}.png".format(psize, level)))

    with open(os.path.join(outfolder, "vocabulary_{}_{}.pkl".format(psize, level)), "wb") as f:
        pickle.dump(vocabulary, f)


def predict_image_bag(image, vocabulary, psize):
    """
    Predict vocabulary on a single image.

    *********************************
    """
    spaceshape = (image.shape[0], image.shape[1])
    positions = unlabeled_regular_grid_list(spaceshape, psize)
    shuffle(positions)
    patches = [image[i:i + psize, j:j + psize].reshape(-1) for i, j in positions]
    y = vocabulary.predict(patches)
    return numpy.sum(y, axis=0)


def predict_imfolder_bags(imfolder, vocabulary, psize):
    """
    Predict vocabulary on slide patches.

    *********************************
    """
    for impath, image in images_in_folder(imfolder, paths=True):
        yield impath, predict_image_bag(image, vocabulary, psize)


def create_bag_dataset(projfolder, outfolder, vocabulary, level, psize, verbose=2):
    """
    Predict bof on an entire slide dataset.

    *********************************
    """
    slide2folder = dataset2folders(projfolder, level)
    k = 0
    outcsv = os.path.join(outfolder, "bow_dataset.csv")
    n = len(vocabulary.cluster_centers_)
    columns = ["Path"]
    columns += ["word_{}".format(c) for c in range(n)]
    with open(outcsv, "w") as csvfile:
        writer = csv.DictWriter(csvfile, columns)
        writer.writeheader()
        for slidename, folder in slide2folder.items():
            k += 1
            if verbose > 0:
                print("processing slide {} / {}".format(k, len(slide2folder)))
                print("translate slide: {}".format(slidename))
                print("translate at level: {}".format(level))
            for impath, bag in predict_imfolder_bags(folder, vocabulary, psize):
                row = {"word_{}".format(c): bag[c] for c in range(len(bag))}
                row["Path"] = impath
                writer.writerow(row)


def learn_bags(bagdataset, outfolder, bagvoclen, psize, level, verbose=2):
    """
    Learn expressions from a dataset of bags.

    *****************************************
    """
    datalist = []
    pathlist = []
    with open(bagdataset, "r") as incsv:
        reader = csv.DictReader(incsv)
        feature_keys = [name for name in reader.fieldnames if "word" in name]
        feature_keys = sorted(feature_keys, key=lambda x: int(x.r_split("_", 1)[-1]))
        for row in reader:
            pathlist.append(row["Path"])
            datalist.append([row[n] for n in feature_keys])
    datalist = numpy.array(datalist)
    expressions = KMeans(n_clusters=bagvoclen)
    preds = expressions.fit_predict(datalist)
    with open(os.path.join(outfolder, "bag_vocabulary_{}_{}.pkl".format(psize, level)), "wb") as f:
        pickle.dump(expressions, f)
    if verbose > 0:
        outdatafolder = os.path.join(outfolder, "DataViz")
        if os.path.isdir(outdatafolder):
            shutil.rmtree(outdatafolder, ignore_errors=True)
        os.makedirs(outdatafolder)

        for k in range(len(expressions.cluster_centers_)):
            centroid_folder = os.path.join(outdatafolder, str(k))
            os.makedirs(centroid_folder)

        idx = 0
        for k in range(len(datalist)):
            image = imread(pathlist[k])
            outdir = os.path.join(outdatafolder, str(preds[k]))
            imsave(os.path.join(outdir, "{}.png".format(str(idx))), image)
