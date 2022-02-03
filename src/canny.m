%******************************************************************************
% DISCLAIMER: All functions used in this program have been referenced and     *
% obtained from mathworks.com examples for the purposes of this assignment.   *
%******************************************************************************

% Canny algorithm parameters.
SIGMA = 1;
LOWER_THRESHOLD = 25;
HIGHER_THRESHOLD = 100;

% Load the image and convert it to a grayscale format.
image = open_and_convert('image1.jfif');

% Retrieve binary image after running the Canny edge detection.
binary_edge_image = canny_edge(image, SIGMA, LOWER_THRESHOLD, HIGHER_THRESHOLD);

% Print the resulting binary image.
figure;
imshow(binary_edge_image)


%******************************************************************************
% Function for loading and formatting the image to the appropriate graysclae 
% image to enhance computational performance.
function bw_image = open_and_convert(image)
    % Load the image provided into the application.
    rgb_image = imread(image);

    % Convert the image from RGB format to grayscale format for better results.
    bw_image = rgb2gray(rgb_image);
end

% Function to implement Canny edge detection and return a binary image.
function binary_image = canny_edge(image, sigma, low_thresh, high_thresh)
    % Variate the size of the Gaussian filter.
    FILTER_SIZE = 5;

    % Create two seperable vectors from the Gaussian filter.
    [G_x, G_y] = create_seperable_gaussian(sigma, FILTER_SIZE);

    % Compute the spatial derivatives using seperated Sobel kernels.
    [I_x, I_y] = sobel_spatial_derivation(image, G_x, G_y);

    % Compute the magnitude and direction of the gradient at every pixel.
    [magnitude, direction] = magnitude_and_direction(I_x, I_y);

    % Get size of the original image.
    [rows, columns] = size(image);

    % Group the gradient directions into the four main ranges.
    direction = classify_direction(direction, rows, columns);

    % Perform non-maximum suppression on the gradient image.
    magnitude = non_maximum_suppression(magnitude, direction, rows, columns);

    % Perform hysterisis thresholding yo obtain binary image containing edges.
    binary_image = hysterisis_thresholding(magnitude, low_thresh, ...
     high_thresh, rows, columns);
end

% Function for generating two 1D Gaussian vectors.
function [G_x, G_y] = create_seperable_gaussian(sigma, filter_size)
    % Construct a normal distribution for the range of the filter.
    positive_end = ceil(filter_size / 2) - 1;
    negative_end = -1 * positive_end;
    range = [negative_end: 1: positive_end];

    % Loop trough the range and substitute it into the Gaussian kernel formula.
    for i = 1:filter_size
        G_x(i) = (1 / (sqrt(2 * pi) * sigma^2)) * exp(-(range(i)^2) / ...
         (2 * (sigma^2)));
    end

    % The Y-direction is simply the transpose of the X-direction.
    G_y = G_x';
end

% Function for computing the spatial derivatives of an image by using seperated
% Sobel kernels convoluted with a 1D Gaussian filter.
function [I_x, I_y] = sobel_spatial_derivation(image, G_x, G_y)
    % Construct 1D Sobel kernels for the horizontal and vertical components.
    % Note the multiplication of these two vectors results in the 2D kernel.
    S_x_x = [-1, 0, 1];
    S_x_y = [1; 2; 1];

    S_y_x = S_x_y';
    S_y_y = -1 .* S_x_x';
    
    % Compute the spatial derivatives by convolving the modified Gaussian 
    % filters with the image in the X and Y directions.
    I_x = matrix_convolution(image, G_x);
    I_y = matrix_convolution(image, G_y);

    % Convolve the Sobel vectors with the Gaussian filter for incorporating
    % a form of edge detection.
    I_x = matrix_convolution(I_x, S_x_x);
    I_x = matrix_convolution(I_x, S_x_y);

    I_y = matrix_convolution(I_y, S_y_x);
    I_y = matrix_convolution(I_y, S_y_y);
end

% Function for computing the convolution of two matricies X and Y.
% Compared results with conv2 function and produced similar results.
function Z = matrix_convolution(X, Y)
    % Obtain the sizes for both matricies.
    [row_X, column_X] = size(X);
    [row_Y, column_Y] = size(Y);

    % Reflect the second matrix about the y axis by rotation 180 degrees.
    h = rot90(Y, 2);

    % Initialize the resultant matrix to be the same size as X.
    Z = zeros(row_X , column_X);

    % Resize the X matrix to comply with the convolution indicies.
    resized_X = convolution_resize(X, h, row_X, column_X, row_Y, column_Y);

    % Compute the convolution summation as described in lecture material.
    for i = 1 : row_X
        for j = 1 : column_X
            for k = 1 : row_Y
                for l = 1 : column_Y
                    Z(i,j) = Z(i,j) + (resized_X(i-1 + k, j-1 + l) * h(k,l));
                end
            end
        end
    end
end

% Function for resizing a matrix in accordance with the another for convolution.
% Any pixels left outside the convolution zone will be padded with zeroes.
function resized = convolution_resize(X, h, row_X, column_X, row_Y, column_Y)
    % Compute the 2 dimensional window size for the convolution to take place.
    % Compute the 2 dimensional h matrix center in ternms or rows and cols.
    middle = floor((size(h)+1)/2);

    % The top and bottom ends are dictated by the rows of h and Y.
    bottom_end = row_Y - middle(1);
    top_end = middle(1) - 1;

    % The left and right ends are dictated by the columns of h and Y.
    right_end = column_Y - middle(2);
    left_end = middle(2) - 1;

    % Concatenate sizes into one parameter each.
    rows_resized = row_X + top_end + bottom_end;
    columns_resized = column_X + left_end + right_end;

    % Reconstruct the X matrix with the additional four corner ends.
    resized = zeros(rows_resized, columns_resized);

    % Copy over the parameters from the original matrix into the resized one.
    % The addition of 1 is to counteract the possibility of index 0.
    for i = top_end + 1 : top_end + row_X
        for j = left_end + 1 : left_end + column_X
            resized(i,j) = X(i-top_end, j-left_end);
        end
    end
end

% Function for computing the gradient magnitude and direction of each pixel.
function [magnitude, direction] = magnitude_and_direction(I_x, I_y)
    magnitude = sqrt((I_x.^2 + I_y.^2));

    % Compute the direction of the gradient and convert it to degrees.
    direction = ((atan2(I_y, I_x)) * 180) / pi;

    % Display the magnitude image as seen in lecture slides.
    figure;
    imshow(magnitude,[]);
end

% Function for distributing the gradient directions into four categories.
function sampled_direction = classify_direction(direction, rows, columns)
    % Loop through the image matrix and assign the gradient directions.
    for i = 1:rows
        for j = 1:columns
            % 0 degree zones on both the upper and lower half of the circle.
            if ((direction(i,j) < 22.5) && (direction(i,j) >= -22.5 ) || ... 
                 (direction(i,j) >= 157.5) && (direction(i,j) < -157.5))
                sampled_direction(i,j) = 0;
            
            % 45 degree zones on both the upper and lower half of the circle.
            elseif ((direction(i,j) >= 22.5) && (direction(i,j) < 67.5) || ...
                 (direction(i,j) >= -157.5) && (direction(i,j) < -112.5))
                sampled_direction(i,j) = 45;

            % 90 degree zones on both the upper and lower half of the circle.
            elseif ((direction(i,j) >= 67.5 && direction(i,j) < 112.5) || ...
                 (direction(i,j) >= -112.5 && direction(i,j) < -67.5))
                sampled_direction(i,j) = 90;

            % 135 degree zones on both the upper and lower half of the circle.
            elseif ((direction(i,j) >= 112.5 && direction(i,j) < 157.5) || ... 
                 (direction(i,j) >= -67.5 && direction(i,j) < -22.5))
                sampled_direction(i,j) = 135;
            end
        end
    end
end

% Function for performing non-maximum suppression along the gradient lines.
function suppressed_magnitude = non_maximum_suppression(magnitude, direction, ...
     rows, columns)
    % Initialize the supressed image with the same values as the original.
    suppressed_magnitude = magnitude;

    % Suppress magnitudes through the rows and columns for the image.
    % Start at index 2 because index 1 cannot support i-1 and j-1.
    for i = 2:rows - 1
        for j = 2:columns - 1
            if (direction(i,j) == 0)
                % Retrieve the max neighbourhood pixel along 0 degree edge.
                neighborhood_max = max([magnitude(i,j), magnitude(i,j+1), ...
                 magnitude(i,j-1)]);

            elseif (direction(i,j) == 45)
                % Retrieve the max neighbourhood pixel along 45 degree edge.
                neighborhood_max = max([magnitude(i,j), magnitude(i+1,j-1), ...
                 magnitude(i-1,j+1)]);

            elseif (direction(i,j) == 90)
                % Retrieve the max neighbourhood pixel along 90 degree edge.
                neighborhood_max = max([magnitude(i,j), magnitude(i+1,j), ...
                 magnitude(i-1,j)]);

            else
                % Retrieve the max neighbourhood pixel along 135 degree edge.
                neighborhood_max = max([magnitude(i,j), magnitude(i+1,j+1), ...
                 magnitude(i-1,j-1)]);
            end

            % Suppress pixel if it was not the highest value along the edge.
            if (neighborhood_max > magnitude(i,j))
                suppressed_magnitude(i,j) = 0;
            end
        end
    end

    % Display the magnitude image as seen in lecture slides.
    figure;
    imshow(suppressed_magnitude,[]);
end

% Function for applying hysteriris thresholding producing a binary image.
function binary_image = hysterisis_thresholding(magnitude, low_thresh, ...
     high_thresh, rows, columns)   
    for i = 2:rows - 1
        for j = 2:columns - 1
            % Suppress pixel if it is below lower threshold.
            if (magnitude(i,j) < low_thresh)
                binary_image(i,j) = 0;
            
            % Keep pixel if it is above higher threshold.
            elseif (magnitude(i,j) > high_thresh)
                binary_image(i,j) = 1;

            % If pixel is in between then inspect neighbouring pixels.
            % Keep pixel if one of these is a strong edge.
            elseif (magnitude(i+1,j) > high_thresh || magnitude(i,j+1) > high_thresh ...
                 || magnitude(i-1,j) > high_thresh  || magnitude(i,j-1) > high_thresh ...
                 || magnitude(i+1,j-1) > high_thresh || magnitude(i-1,j+1) > high_thresh ...
                 || magnitude(i+1,j+1) > high_thresh || magnitude(i-1,j-1) > high_thresh)
                binary_image(i,j) = 1;
            
            % Suppress all other pixels in the middle ground.
            else
                binary_image(i,j) = 0;
            end
        end
    end
end
