"""
Main file for the cv_mosaicing project
"""

import cv2
import argparse
import numpy as np

###################
# Main CV functions
###################


def get_images(image_paths, scale_factor=1.0):
    """
    Read in the images

    ## Returns:
        images: a list of the images
    """
    images = []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        if scale_factor != 1.0:
            img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
        images.append(img)
    return images


def get_nonmax_suppression(img, window_size=5):
    """
    Apply non-maximum suppression to an image

    ## Returns:
        img_copy: a copy of the image with non-maximum suppression applied
    """
    img_copy = img.copy()
    img_min = img.min()

    for r, c in np.ndindex(img_copy.shape):
        # get window around specific pixel
        c_lower = max(0, c-window_size//2)
        c_upper = min(img_copy.shape[1], c+window_size//2)
        r_lower = max(0, r-window_size//2)
        r_upper = min(img_copy.shape[0], r+window_size//2)
        
        # set pixel to img_min so it is not included in max calculation
        temp = img_copy[r, c]
        img_copy[r, c] = img_min

        # if pixel is the max in the window, keep it, otherwise keep it img_min
        if temp > img_copy[r_lower:r_upper, c_lower:c_upper].max():
            img_copy[r, c] = temp
    
    return img_copy


def get_harris_corners(img, num_corners=400, window_size=5, neighborhood_size=7):
    """
    Detect Harris corners in an image, returning their locations and neighborhoods

    ## Returns:
        corners: (num_corners, 2) array of (x, y) coordinates of the corners
        neighborhood: (num_corners, neighborhood_size, neighborhood_size) array of the neighborhoods around the corners
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

    # Calculate Non-maximum suppression
    Rs = get_nonmax_suppression(R)

    # Display the images in a single window (debugging)
    Ixx_disp = cv2.normalize(Ixx, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    Iyy_disp = cv2.normalize(Iyy, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    R_disp = cv2.normalize(R, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    Rs_disp = cv2.normalize(Rs, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    Hor1 = cv2.hconcat([Ixx_disp, Iyy_disp])
    Hor2 = cv2.hconcat([R_disp, Rs_disp])
    cv2.imshow("Harris Corner Detection (Gradient and Response visualization)", cv2.vconcat([Hor1, Hor2]))
    

    # Exception if the number of corners is greater than the number of pixels in the image
    if num_corners > Rs.size:
        raise ValueError("num_corners must be less than the number of pixels in the image")
    # Return the top num_corners corners by sorting and returning the indices
    corners = np.unravel_index(np.argsort(Rs, axis=None)[-num_corners:], R.shape)
    corners = np.stack((corners[1], corners[0]), axis=1)

    # Get the neighborhoods of the corners
    neighborhoods = np.empty((num_corners, neighborhood_size, neighborhood_size))
    for i, (x, y) in enumerate(corners):
        # TODO handle when too close to the edge
        neighborhoods[i] = img_gray[y-neighborhood_size//2:y+neighborhood_size//2+1, x-neighborhood_size//2:x+neighborhood_size//2+1]

    return corners, neighborhoods


def get_correspondences(corners1, neighborhoods1, corners2, neighborhoods2):
    """
    Find correspondences between the two images, returned as a dictionary mapping the corners
    from image1 to the corners in image2
    
    ## Returns:
        final_correspondences: dictionary mapping (x1, y1) to (x2, y2)
    """
    final_correspondences = {}
    
    # normalize neighborhoods
    neighborhoods1 = (neighborhoods1 - neighborhoods1.mean(axis=0, keepdims=True))
    neighborhoods1 = neighborhoods1 / np.linalg.norm(neighborhoods1, axis=0)

    neighborhoods2 = (neighborhoods2 - neighborhoods2.mean(axis=0, keepdims=True))
    neighborhoods2 = neighborhoods2 / np.linalg.norm(neighborhoods2, axis=0)

    for c1, n1 in zip(corners1, neighborhoods1):
        best_corner = (corners2[0], 0)
        for c2, n2 in zip(corners2, neighborhoods2):
            corr = np.sum(n1 * n2)
            if corr > best_corner[1]:
                best_corner = (c2, corr)
        
        final_correspondences[tuple(c1)] = best_corner[0]

    return final_correspondences


def calculate_homography(points1, points2):
    """
    Estimate the homography between two sets of 4 corners using the 8-point algorithm
    ## Returns:
        homography: 3x3 homography matrix
    """
    # Create A matrix
    A = np.zeros((8, 9))
    for i in range(4):
        x1, y1 = points1[i]
        x2, y2 = points2[i]
        # Create A matrix for each point
        A[2*i] = [x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2]
        A[2*i+1] = [0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2]
    # Solve for h using SVD
    U, S, V = np.linalg.svd(A.T @ A)
    h = V[-1]
    # Reshape h into a 3x3 matrix
    homography = h.reshape((3, 3))
    # Normalize homography
    homography = homography / homography[2, 2]

    return homography


def homography_ransac(correspondences):
    """
    Estimate the homography between the two images using the given correspondences

    ## Returns: 
        homography: 3x3 homography matrix
    """
    # Convert correspoindence dictionary to (N, 2 x 2) array
    correspondences = np.array(list(correspondences.items()))
    # Implement RANSAC for homography estimation
    max_iterations = 1000
    best_homography = None
    max_inliers = 0
    best_set_correspondences = {}
    temp_correspondences = {}
    inliner_threshold = 1
    iter = 0
    while (iter <= max_iterations):
        # Sample 4 points from the correspondences
        sample = np.random.choice(correspondences.shape[0], 4, replace=False)
        sample = correspondences[sample]
        # Calculate homography
        homography = calculate_homography(sample[:, 0], sample[:, 1])
        # Calculate inliers
        inliers = 0
        for c in correspondences:
            # Get the points
            p1 = np.array([c[0][0], c[0][1], 1])
            p2 = np.array([c[1][0], c[1][1], 1])
            # Perform the homography transformation
            p2_hat = np.dot(homography, p1)
            # Normalize
            p2_hat = np.isfinite(p2_hat / p2_hat[2]).all()
            if np.linalg.norm(p2 - p2_hat) < inliner_threshold:
                inliers += 1
                temp_correspondences[tuple(c[0])] = c[1]
        # Check if the homography is better than the previous one
        if (inliers > max_inliers):
            print(inliers)
            max_inliers = inliers
            best_homography = homography
            best_set_correspondences = temp_correspondences
        temp_correspondences = {}
        iter += 1

    # Return homography
    return best_homography, best_set_correspondences

def warp_and_blend(img1, img2, homography):
    """
    Warp one image onto the other one, blending overlapping pixels together to create
    a single image that shows the union of all pixels from both input images

    ## Returns:
        output: the blended image

    """
    left_img = img1.copy()
    right_img = img2.copy()
    # Check if img2 is to the left of img1
    if homography[0, 2] < 0:
        # Swap the images
        left_img, right_img = right_img, left_img
        homography = np.linalg.inv(homography)

    result = cv2.warpPerspective(right_img, homography, 
                                   (left_img.shape[1] + right_img.shape[1], left_img.shape[0]))
    
    result[0:left_img.shape[0], 0:left_img.shape[1]] = left_img
    return result


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
    for (c1r, c1c), (c2r, c2c) in correspondences.items():
        cv2.line(images, (c1r, c1c), (c2r+img1.shape[1], c2c), np.random.randint(20, 255, 3).tolist())
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
    # image, and then do non-maximum suppression to get a sparse set of corner features.
    corners1, neighborhoods1 = get_harris_corners(img1, num_corners=250, neighborhood_size=19)
    corners2, neighborhoods2 = get_harris_corners(img2, num_corners=250, neighborhood_size=19)
    display_harris_corners(img1, corners1, img2, corners2)

    # iii. Find correspondences between the two images: given two set of corners from the
    # two images, compute normalized cross correlation (NCC) of image patches centered
    # at each corner. (Note that this will be O(n2) process.) Choose potential corner
    # matches by finding pair of corners (one from each image) such that they have the
    # highest NCC value. You may also set a threshold to keep only matches that have a
    # large NCC score.
    correspondences = get_correspondences(corners1, neighborhoods1, corners2, neighborhoods2)
    #display_correspondences(img1, img2, correspondences)

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
    homography, best_set_corresp = homography_ransac(correspondences)
    print("Homography: \n", homography)
    display_correspondences(img1, img2, best_set_corresp)
    # homography = np.array([[0.8, 0, 200],
    #                        [0,  0.8,0],
    #                        [0,  0,  1]])

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
    #output = warp_and_blend(img1, img2, homography)

    # Save and display the output image
    #cv2.imwrite("output_final.jpg", output)
    #cv2.imshow("output final", output)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()

