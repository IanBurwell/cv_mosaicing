"""
Main file for the cv_mosaicing project
"""

import cv2
import numpy as np
import argparse


###################
# Main CV functions
###################


def get_images(image_paths, scale_factor=1.0):
    """
    Read in the images
    """
    images = []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        if scale_factor != 1.0:
            img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
        images.append(img)
    return images


def get_harris_corners(img, num_corners=200, window_size=5, neighborhood_size=7):
    """
    Detect Harris corners in an image, returning their locations and neighborhoods
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # calulate derivatives
    Ix = cv2.Sobel(img_gray, ddepth=-1, dx=1, dy=0, ksize=3)
    Iy = cv2.Sobel(img_gray, ddepth=-1, dx=0, dy=1, ksize=3)

    # derivative products
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # sum of products 
    sum_kernel = np.ones((window_size, window_size))
    Sxx = cv2.filter2D(src=Ixx, ddepth=-1, kernel=sum_kernel)
    Syy = cv2.filter2D(src=Iyy, ddepth=-1, kernel=sum_kernel)
    Sxy = cv2.filter2D(src=Ixy, ddepth=-1, kernel=sum_kernel)

    # calculate C matricies and R values
    R = np.empty(shape=Sxx.shape, dtype=np.float32)
    for i, j in np.ndindex(Sxx.shape):
        # set edges to zero as we cannot give them features easily
        if i < neighborhood_size//2 or i >= (R.shape[0] - neighborhood_size//2) or j < neighborhood_size//2 or j >= (R.shape[1] - neighborhood_size//2):
            R[i, j] = 0
            continue
        # calculate R value
        C = np.array([[Sxx[i, j], Sxy[i, j]], [Sxy[i, j], Syy[i, j]]])
        R[i, j] = np.linalg.det(C) - 0.04 * (np.trace(C) ** 2)

    cv2.imshow("R",  cv2.normalize( R, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
    cv2.imshow("Iyx", cv2.normalize(Ixx, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
    cv2.imshow("Iyy", cv2.normalize(Iyy, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))

    # Calculate Non-maximum suppression
    # TODO
    # idea: blur and look for local maxima

    # Return the top num_corners corners by sorting and returning the indices
    corners = np.unravel_index(np.argsort(R, axis=None)[-num_corners:], R.shape)
    corners = np.stack((corners[1], corners[0]), axis=1)

    # Get the neighborhoods of the corners
    neighborhoods = np.empty((num_corners, neighborhood_size, neighborhood_size))
    for i, (x, y) in enumerate(corners):
        # TODO handle when too close to the edge
        neighborhoods[i] = R[y-neighborhood_size//2:y+neighborhood_size//2+1, x-neighborhood_size//2:x+neighborhood_size//2+1]

    return corners, neighborhoods


def get_correspondences(img1, corners1, neighborhoods1, img2, corners2, neighborhoods2):
    """
    Find correspondences between the two images, returned as a dictionary mapping the corners
    from image1 to the corners in image2
    """
    correspondences = {}
    # TODO
    return correspondences


def estimate_homography(corners1, corners2, correspondences):
    """
    Estimate the homography between the two images using the given correspondences
    """
    homography = np.identity(3)
    # TODO
    return homography


def warp_and_blend(img1, img2, homography):
    """
    Warp one image onto the other one, blending overlapping pixels together to create
    a single image that shows the union of all pixels from both input images
    """
    output = np.zeros_like(img1)
    # TODO
    return output


##########################
# Display helper functions
##########################


def display_harris_corners(img1, corners1, img2=None, corners2=None):
    """
    Display the Harris corners on top of the image
    """
    img1_copy = img1.copy()
    for corner in corners1:
        cv2.circle(img1_copy, corner, 2, (0, 0, 255), -1)
    if img2 is not None:
        img2_copy = img2.copy()
        for corner in corners2:
            cv2.circle(img2_copy, corner, 2, (0, 0, 255), -1)
        cv2.imshow("harris corners", np.concatenate((img1_copy, img2_copy), axis=1))
        cv2.imwrite("output_harris_corners.jpg", np.concatenate((img1_copy, img2_copy), axis=1))
    else:
        cv2.imshow("harris corners", img1_copy)
        cv2.imwrite("output_harris_corners.jpg", img1_copy)


def display_correspondences(img1, img2, correspondences):
    """
    Display the correspondences between the two images
    """
    images = np.concatenate((img1, img2), axis=1) 
    for (c1x, c1y), (c2x, c2y) in correspondences.items():
        cv2.line(images, (c1x, c1y), (c2x, c2y+img1.shape[0]), (0, 0, 255))
    cv2.imshow("correspondences", images)
    cv2.imwrite("output_correspondences.jpg", images)


######
# Main 
######


def get_args():
    """
    Get command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("image1", help="Path to the first image")
    parser.add_argument("image2", help="Path to the second image")
    return parser.parse_args()


def main():
    # Read in the command line arguments
    args = get_args()

    # Read in two images. (Note: if the images are large, you may want to reduce their
    # size to keep running time reasonable! Document in your report the scale factor you
    # used.)
    img1, img2 = get_images([args.image1, args.image2], scale_factor=0.5)
    cv2.imwrite("output_inputs.jpg", np.concatenate((img1, img2), axis=1))
    cv2.imshow("input images", np.concatenate((img1, img2), axis=1))

    # ii. Apply Harris corner detector to both images: compute Harris R function over the
    # image, and then do non-maimum suppression to get a sparse set of corner features.
    corners1, neighborhoods1 = get_harris_corners(img1)
    corners2, neighborhoods2 = get_harris_corners(img2)
    display_harris_corners(img1, corners1, img2, corners2)

    # iii. Find correspondences between the two images: given two set of corners from the
    # two images, compute normalized cross correlation (NCC) of image patches centered
    # at each corner. (Note that this will be O(n2) process.) Choose potential corner
    # matches by finding pair of corners (one from each image) such that they have the
    # highest NCC value. You may also set a threshold to keep only matches that have a
    # large NCC score.
    correspondences = get_correspondences(img1, corners1, neighborhoods1, img2, corners2, neighborhoods2)
    display_correspondences(img1, img2, correspondences)

    # iv. Estimate the homography using the above correspondences. Note that these cor-
    # respondences are likely to have many errors (outliers). That is ok: you should use
    # RANSAC to robustly estimate the homography from the noisy correspondences:
    # A. Repeatedly sample minimal number of points needed to estimate a homography
    # (4 pts in this case).
    # B. Compute a homography from these four points.
    # C. Map all points using the homagraphy and comparing distances between pre-
    # dicted and observed locations to determine the number of inliers.
    # D. At the end, compute a least-squares homgraphy from ALL the inliers in the
    # largest set of inliers.
    homography = estimate_homography(corners1, corners2, correspondences)

    # v. Warp one image onto the other one, blending overlapping pixels together to create
    # a single image that shows the union of all pixels from both input images. You can
    # choose which of the images to warp. The steps are as follows:
    # A. Determine how big to make the final output image so that it contains the union
    # of all pixels in the two images.
    # B. Copy the image that does not have to be warped into the appropriate location
    # in the output.
    # C. Warp the other image into the output image based on the estimated homography
    # (or its inverse). You can use matlab functions if you want or write your own
    # warping function.
    # D. Use any of the blending schemes we will discuss in class to blend pixels in the
    # area of overlap between both images.
    output = warp_and_blend(img1, img2, homography)

    # Save and display the output image
    cv2.imwrite("output_final.jpg", output)
    cv2.imshow("output final", output)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()

