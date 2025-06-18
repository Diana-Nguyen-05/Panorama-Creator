%-------------------------------------------------------------------------
% Functions
%-------------------------------------------------------------------------

% 2
% FAST
function features = my_fast_detector(img, threshold, n)
    % convert to gray if needed
    img = im2gray(img);

    % detected features
    features = [];
    
    % get bresenham circle (by using offsets from pixel)
    % i.e. 0 3 means 3 up from pixel, etc...
    circle = [0 3; 1 3; 2 2; 3 1; 3 0; 3 -1; 2 -2; 1 -3; 0 -3; -1 -3; -2 -2; -3 -1; -3 0; -3 1; -2 2; -1 3];

    % get num of rows and columns
    [rows, cols] = size(img);

    % keep track of light and dark
    % create matrix of zeros to avoid resizing later on
    bright = zeros(rows,cols);
    dark = zeros(rows,cols);

    % shift image instead of looping
    for i=1: 16
        shift_x = circle(i,1);
        shift_y = circle(i,2);

        % use circshift to handle edge cases too
        shifted = circshift(img, [shift_y, shift_x]);

        % compare
        bright = bright + (shifted > img + threshold);
        dark = dark + (shifted < img - threshold);
    end

    % check for contiguity 
    % check each pixel
    for y=4: rows-4
        for x=4: cols-4
            cur_pixel = img(y,x);
            % only check if n is possibly reached
            if bright(y,x) >= n || dark(y,x) >= n
                % initialize to false
                corner = false;

                % check for consecutive n
                for i=1:16
                    bright_in_row = 0;
                    dark_in_row = 0;
                    % check all possible n consecutives in a row
                    for j=0: n-1
                        if i+j > size(circle,1)
                            off_x = x + circle(i+j-16,1);
                            off_y = y + circle(i+j-16,2);
                        else
                            off_x = x + circle(i+j, 1);
                            off_y = y + circle(i+j, 2);
                        end

                        % compare
                        if img(off_y, off_x) > cur_pixel + threshold
                            bright_in_row = bright_in_row + 1;
                        elseif img(off_y, off_x) < cur_pixel - threshold
                            dark_in_row = dark_in_row + 1;
                        else % did not have n consecutive
                            bright_in_row = 0;
                            dark_in_row = 0;
                            break;
                        end
                    end
                    % if is n consecutive, consider it a corner
                    if bright_in_row == n || dark_in_row == n
                        corner = true;
                        break;
                    end
                end
                % if is a coner, add to features
                if corner
                    features = [features; x,y];
                end
            end
        end
    end
end

%-------------------------------------------------------------------------
%-------------------------------------------------------------------------

% 2 FASTR
% function that gets harris cornerness meaure
function FRfeatures = harrisCornerness(features, img, threshold)
    % turn img to grey and double
    img = im2double(im2gray(img));

   % get gradients
   sobelX = [-1 0 1; -2 0 2; -1 0 1];
   sobelY = sobelX';
   Ix = imfilter(img, sobelX);
   Iy = imfilter(img, sobelY);

   % matrix
   Ix2 = Ix.^2;
   Iy2 = Iy.^2;
   Ixy = Ix .* Iy;
   % apply gaussian blurring to reduce noise
   g = fspecial('gaussian', 5, 1);
   Ix2 = imfilter(Ix2, g);
   Iy2 = imfilter(Iy2, g);
   Ixy = imfilter(Ixy, g);
   
   % cornerness measure
   k = 0.04;
   response = (Ix2 .* Iy2 -  Ixy.^2) - k * (Ix2 + Iy2).^2;

   % selection
   FRfeatures = [];
   for i=1:size(features,1)
       x = features(i,1);
       y = features(i,2);
       if response(y,x) > threshold
           FRfeatures = [FRfeatures; x,y];
       end
   end
end

%-------------------------------------------------------------------------
%-------------------------------------------------------------------------

% descriptor
function [matchedP1, matchedP2, matchedPairs] = descriptor(img1, img2, features1, features2)
    % make imgs grey
    img1 = im2gray(img1);
    img2 = im2gray(img2);
       
    % convert features to SURF points
    surfP1 = SURFPoints(features1);
    surfP2 = SURFPoints(features2);

    % FREAK
    [desc1, valid_points1] = extractFeatures(img1, surfP1,'Method', 'FREAK');
    [desc2, valid_points2] = extractFeatures(img2, surfP2,'Method', 'FREAK');

    % match!!
    matched = matchFeatures(desc1,desc2);

    % get matched points
    matched1s = valid_points1(matched(:,1));
    matched2s = valid_points2(matched(:,2));

    % put into x,y list
    matched1 = [];
    matched2 = [];
    for i=1:matched1s.size
        matched1(i,:) = matched1s(i).Location;
        matched2(i,:) = matched2s(i).Location;
    end

    % returns
    matchedP1 = matched1;
    matchedP2 = matched2;
    matchedPairs = matched;

end

%-------------------------------------------------------------------------
%-------------------------------------------------------------------------

function panorama = PanoramaMaker(ImgSet, tforms)
    % get the number of imahes
    numImg = numel(ImgSet);

    % initialize the panorama
    imgSize = zeros(numImg,2);
    xlim = zeros(numImg,2);
    ylim = zeros(numImg,2);

    for i=1:numImg
        imgSize(i,:) = size(ImgSet{i}(:,:,1));
        [xlim(i,:),ylim(i,:)] = outputLimits(tforms(i),[1 imgSize(i,2)],[1 imgSize(i,1)]);
    end

    maxImgSize = max(imgSize);

    % find min/max limits
    xMin = min([1; xlim(:)]);
    xMax = max([maxImgSize(2); xlim(:)]);
    yMin = min([1; ylim(:)]);
    yMax = max([maxImgSize(1); ylim(:)]);

    width = round(xMax-xMin);
    height = round(yMax-yMin);

    % blank panorama
    panorama = zeros([height width 3], 'like', ImgSet{1});

    % panorama view
    panoramaView = imref2d([height width],[xMin xMax],[yMin yMax]);

    % stitch
    for i=1:numImg
        img = ImgSet{i};

        warpedImage = imwarp(img, tforms(i), 'OutputView',panoramaView);
        mask = imwarp(true(size(img,1), size(img,2)), tforms(i), 'OutputView',panoramaView);
        panorama = imblend(warpedImage, panorama, mask, foregroundopacity=1);

    end
end

%-------------------------------------------------------------------------
%-------------------------------------------------------------------------

% main

% 1
% read and prepare image sets
% set 1
S1_img1 = imread('S1-im1.png'); % 750x480
S1_img2 = imread('S1-im2.png'); % 750x480

% set 2
S2_img1 = imread('S2-im1.png');
S2_img2 = imread('S2-im2.png');
S2_img1 = imresize(S2_img1,[750,480]);
S2_img2 = imresize(S2_img2, [750,480]);

% set 3
S3_img1 = imread('S3-im1.png');
S3_img2 = imread('S3-im2.png');
S3_img3 = imread('S3-im3.png');
S3_img4 = imread('S3-im4.png');

S3_img1 = imresize(S3_img1, [750,480]);
S3_img2 = imresize(S3_img2, [750,480]);
S3_img3 = imresize(S3_img3, [750,480]);
S3_img4 = imresize(S3_img4, [750,480]);

% set 4
S4_img1 = imread('S4-im1.png');
S4_img2 = imread('S4-im2.png');
S4_img3 = imread('S4-im3.png');
S4_img4 = imread('S4-im4.png');

S4_img1 = imresize(S4_img1, [750,480]);
S4_img2 = imresize(S4_img2, [750,480]);
S4_img3 = imresize(S4_img3, [750,480]);
S4_img4 = imresize(S4_img4, [750,480]);

%-------------------------------------------------------------------------

% 2 FAST
fast_time = zeros(1,4);

% set 1
tic;
features_S1_im1 = my_fast_detector(S1_img1, 30, 12);
fast_time(1) = toc;
tic;
features_S1_im2 = my_fast_detector(S1_img2, 30, 12);
fast_time(2) = toc;
% set 2
tic;
features_S2_im1 = my_fast_detector(S2_img1, 20, 12);
fast_time(3) = toc;
tic;
features_S2_im2 = my_fast_detector(S2_img2, 20, 12);
fast_time(4) = toc;

avg_fast = mean(fast_time);

% save FAST images 
% set 1
figure;
imshow(S1_img1);
hold on;
plot(features_S1_im1(:,1),features_S1_im1(:,2), 'ro', 'MarkerSize',2,'LineWidth',1);
hold off;
saveas(gcf, 'S1-fast.png');
close;

% set 2
figure;
imshow(S2_img1);
hold on;
plot(features_S2_im1(:,1),features_S2_im1(:,2), 'ro', 'MarkerSize',2,'LineWidth',1);
hold off;
saveas(gcf, 'S2-fast.png');
close;

%-------------------------------------------------------------------------

% FASTR
fastR_time = zeros(1,4);

% set 1
tic;
FRfeatures_S1_im1 = harrisCornerness(features_S1_im1, S1_img1, 0.07);
fastR_time(1) = toc;
tic;
FRfeatures_S1_im2 = harrisCornerness(features_S1_im2, S1_img2, 0.07);
fastR_time(2) = toc;
% set 2
tic;
FRfeatures_S2_im1 = harrisCornerness(features_S2_im1, S2_img1, 0.05);
fastR_time(3) = toc;
tic;
FRfeatures_S2_im2 = harrisCornerness(features_S2_im2, S2_img2, 0.05);
fastR_time(4) = toc;

% get average
avg_fastR = mean(fastR_time);

% calculate difference
time_diff = avg_fastR - avg_fast;

% save FASTR images
% set 1
figure;
imshow(S1_img1);
hold on;
plot(FRfeatures_S1_im1(:,1),FRfeatures_S1_im1(:,2), 'ro', 'MarkerSize', 2, 'LineWidth', 1);
hold off;
saveas(gcf, 'S1-fastR.png');
close;
% set 2
figure;
imshow(S2_img1);
hold on;
plot(FRfeatures_S2_im1(:,1),FRfeatures_S2_im1(:,2), 'ro', 'MarkerSize', 2, 'LineWidth', 1);
hold off;
saveas(gcf, 'S2-fastR.png');
close;

% it significantly reduced the number of weak points I had before and seems
% to have kept points with stronger and sharper edges/corners in places
% where intensity change is more prominent 

%-------------------------------------------------------------------------

% 3 descriptors - FREAK
% FAST
fastDesc_time = zeros(1,2);

% set 1
tic;
[s1MatchP1F, s1MatchP2F, s1MatchedPairsF] = descriptor(S1_img1, S1_img2, features_S1_im1, features_S1_im2);
fastDesc_time(1) = toc;
% set 2
tic;
[s2MatchP1F, s2MatchP2F, s2MatchedPairsF] = descriptor(S2_img1, S2_img2, features_S2_im1, features_S2_im2);
fastDesc_time(2) = toc;

% get average
avg_descTime = mean(fastDesc_time);

% FASTR
fastRdesc_time = zeros(1,2);

% set 1
tic;
[s1MatchP1FR, s1MatchP2FR, s1MatchedPairsFR] = descriptor(S1_img1, S1_img2, FRfeatures_S1_im1, FRfeatures_S1_im2);
fastRdesc_time(1) = toc;
% set 2
tic;
[s2MatchP1FR, s2MatchP2FR, s2MatchedPairsFR,] = descriptor(S2_img1, S2_img2, FRfeatures_S2_im1, FRfeatures_S2_im2);
fastRdesc_time(2) = toc;

% get average
avg_descTimeR = mean(fastRdesc_time);

% calculate difference
desc_time_diff = avg_descTimeR - avg_descTime;

% save descriptors
% set 1
figure;
showMatchedFeatures(S1_img1, S1_img2, s1MatchP1F, s1MatchP2F, 'Montage');
saveas(gcf,'S1-fastMatch.png');
close;

figure;
showMatchedFeatures(S1_img1, S1_img2, s1MatchP1FR, s1MatchP2FR, 'Montage');
saveas(gcf,'S1-fastRMatch.png');
close;

% set 2
figure;
showMatchedFeatures(S2_img1, S2_img2, s2MatchP1F, s2MatchP2F, 'Montage');
saveas(gcf,'S2-fastMatch.png');
close;

figure;
showMatchedFeatures(S2_img1, S2_img2, s2MatchP1FR, s2MatchP2FR, 'Montage');
saveas(gcf,'S2-fastRMatch.png')
close;

%-------------------------------------------------------------------------

% 4 RANSAC and panoramas
% although it's probably not good to do, I wanted to try my own kind of
% take, so i made a panorama function, but it doesn't work with more than 2
% images, trust me I tried, but I can't get the warps for 3 and 4 to
% properly function (most likely accumulated wrong) so I relied more on the
% tutorial code to get through 4 images

% FAST RANSAC configurations
maxFastTrials = 200;
fastConf = 99;
fastInlier = 3;

% FASTR RANSAC configurations
maxFastrTrials = 180;
fastrConf = 99;
fastrInlier = 3;

% set 1
% FASTR
S1ImgSet = {S1_img1, S1_img2};

S1tformFASTR = estgeotform2d(s1MatchP2FR, s1MatchP1FR,'projective','MaxNumTrials',...
    maxFastrTrials,'Confidence',fastrConf,'MaxDistance',fastrInlier);
S1tformsR = [eye(3), S1tformFASTR];
S1panorama = PanoramaMaker(S1ImgSet, S1tformsR);
% save
%imshow(S1panorama);
imwrite(S1panorama, 'S1-panorama.png');

% FAST
S1tformFAST = estgeotform2d(s1MatchP2F,s1MatchP1F,'projective','MaxNumTrials',...
    maxFastTrials,'Confidence',fastConf,'MaxDistance',fastInlier);
S1tforms = [eye(3), S1tformFAST];
S1panorama = PanoramaMaker(S1ImgSet, S1tforms);

% set 2
S2tformFAST = estgeotform2d(s2MatchP2F,s2MatchP1F,'projective','MaxNumTrials',...
    maxFastTrials,'Confidence',fastConf,'MaxDistance',fastInlier);
S2tformFASTR = estgeotform2d(s2MatchP2FR, s2MatchP1FR,'projective','MaxNumTrials',...
    maxFastrTrials,'Confidence',fastrConf,'MaxDistance',fastrInlier);

S2ImgSet = {S2_img1, S2_img2};
S2tforms = [eye(3), S2tformFASTR];
S2panorama = PanoramaMaker(S2ImgSet, S2tforms);

% save
%imshow(S2panorama);
imwrite(S2panorama, 'S2-panorama.png');

% set 3
% get features
features_S3_im1 = my_fast_detector(S3_img1, 30, 12);
features_S3_im2 = my_fast_detector(S3_img2, 30, 12);

% cornerness measure
FRfeatures_S3_im1 = harrisCornerness(features_S3_im1, S3_img1, 0.07);
FRfeatures_S3_im2 = harrisCornerness(features_S3_im2, S3_img2, 0.07);

% match features
[s3MatchP1FR, s3MatchP2FR, s3MatchedPairsFR] = descriptor(S3_img1, S3_img2, FRfeatures_S3_im1, FRfeatures_S3_im2);

% stitch together
S3tformFASTR = estgeotform2d(s3MatchP2FR, s3MatchP1FR,'projective','MaxNumTrials',...
    maxFastrTrials,'Confidence',fastrConf,'MaxDistance',fastrInlier);

S3ImgSet = {S3_img1, S3_img2};
S3tforms = [eye(3), S3tformFASTR];
S32panorama = PanoramaMaker(S3ImgSet, S3tforms);

% save
imshow(S32panorama);
imwrite(S32panorama,'S3-panorama.png');

% set 4
% get features
features_S4_im1 = my_fast_detector(S4_img1, 30, 12);
features_S4_im2 = my_fast_detector(S4_img2, 30, 12);

% cornerness measure
FRfeatures_S4_im1 = harrisCornerness(features_S4_im1, S4_img1, 0.07);
FRfeatures_S4_im2 = harrisCornerness(features_S4_im2, S4_img2, 0.07);

% match features
[s4MatchP1FR, s4MatchP2FR, s4MatchedPairsFR] = descriptor(S4_img1, S4_img2, FRfeatures_S4_im1, FRfeatures_S4_im2);

% stitch together
S4tformFASTR = estgeotform2d(s4MatchP2FR, s4MatchP1FR,'projective','MaxNumTrials',...
    maxFastrTrials,'Confidence',fastrConf,'MaxDistance',fastrInlier);

S4ImgSet = {S4_img1, S4_img2};
S4tforms = [eye(3), S4tformFASTR];
S42panorama = PanoramaMaker(S4ImgSet, S4tforms);

% save
%imshow(S42panorama);
imwrite(S42panorama,'S4-panorama.png');

%-------------------------------------------------------------------------

% 5 - stitch 4 images instead of two
% after a week of debugging and recoding and staring at the tutorial code,
% this is all I've got
% probably could have made this one function for them all, but I had a
% midterm. Honestly my fault

% set 3
% not the best image set, but I got so frustrated so this is what I'm
% working with
S3_imgSet = {S3_img1, S3_img2, S3_img3, S3_img4};

% number of images in set
% initialize
numImages = numel(S3_imgSet);
tforms(numImages) = projtform2d;
imageSize = zeros(numImages, 2);

% Detect and extract features for the first image
grayImage1 = im2gray(S3_imgSet{1});
S3points1 = my_fast_detector(grayImage1, 30, 12);
S3pointsR1 = harrisCornerness(S3points1, grayImage1, 0.01);
[featuresPrevious, pointsPrevious] = extractFeatures(grayImage1, S3pointsR1);

imageSize(1, :) = size(grayImage1);

% go thorugh rest of the images
for n = 2:numImages
    img = S3_imgSet{n};
    grayImage = im2gray(img);
    imageSize(n, :) = size(grayImage);

    % Detect and extract features for the current image
    S3points = my_fast_detector(grayImage, 30, 12);
    S3pointsR = harrisCornerness(S3points, grayImage, 0.01);
    [features, pointsR] = extractFeatures(grayImage, S3pointsR);
    
    % find matches (my match function doesnt work cause mine takes in 2
    % images as input)
    indexPairs = matchFeatures(features, featuresPrevious);
    matchedPoints = pointsR(indexPairs(:, 1), :);
    matchedPointsPrev = pointsPrevious(indexPairs(:, 2), :);

    % accumulate transformations
    tforms(n) = estgeotform2d(matchedPoints, matchedPointsPrev, ...
        'projective', 'MaxNumTrials', maxFastrTrials, 'Confidence', fastrConf, 'MaxDistance', fastrInlier);
        

    % make transformation relative to the first image
    tforms(n).A = tforms(n - 1).A * tforms(n).A;

    % Update features for the next iteration
    featuresPrevious = features;
    pointsPrevious = pointsR;
end    

% initialize x and y limits on transformations
xlim = zeros(numImages, 2);
ylim = zeros(numImages, 2);

% limits
for i = 1:numel(tforms)           
    [xlim(i,:),ylim(i,:)] = outputLimits(tforms(i),[1 imageSize(i,2)],[1 imageSize(i,1)]);    
end

% initialize the panorama
for i = 1:numel(tforms)           
    [xlim(i,:),ylim(i,:)] = outputLimits(tforms(i),[1 imageSize(i,2)],[1 imageSize(i,1)]);
end
maxImageSize = max(imageSize);

% get the output limits
xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);
yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);

width  = round(xMax - xMin);
height = round(yMax - yMin);

% blank canvas for panorama
S3panorama = zeros([height width 3],"like",img);

% size of panorama
panoramaView = imref2d([height width],[xMin xMax],[yMin yMax]);

% make the panorama!!!!!!!!!!!!!!!!!!
for i = 1:numImages
    img = S3_imgSet{i};   
    warpedImage = imwarp(img,tforms(i),'OutputView',panoramaView);                 
    mask = imwarp(true(size(img,1),size(img,2)),tforms(i),'OutputView',panoramaView);
    S3panorama = imblend(warpedImage,S3panorama,mask,foregroundopacity=1);
end

% save
imshow(S3panorama)
imwrite(S3panorama, 'S1-largepanorama.png')



% set 4
% make a set for the images used in this well.. set
S4_imgSet = {S4_img1, S4_img2, S4_img3, S4_img4};

% number of images in set
% initialize
numImages = numel(S4_imgSet);
tforms(numImages) = projtform2d;
imageSize = zeros(numImages, 2);

% Detect and extract features for the first image
grayImage1 = im2gray(S4_imgSet{1});
S4points1 = my_fast_detector(grayImage1, 30, 12);
S4pointsR1 = harrisCornerness(S4points1, grayImage1, 0.01);
[featuresPrevious, pointsPrevious] = extractFeatures(grayImage1, S4pointsR1);

imageSize(1, :) = size(grayImage1);

% go thorugh rest of the images
for n = 2:numImages
    img = S4_imgSet{n};
    grayImage = im2gray(img);
    imageSize(n, :) = size(grayImage);

    % Detect and extract features for the current image
    S4points = my_fast_detector(grayImage, 30, 12);
    S4pointsR = harrisCornerness(S4points, grayImage, 0.01);
    [features, pointsR] = extractFeatures(grayImage, S4pointsR);
    
    % find matches (my match function doesnt work cause mine takes in 2
    % images as input)
    indexPairs = matchFeatures(features, featuresPrevious);
    matchedPoints = pointsR(indexPairs(:, 1), :);
    matchedPointsPrev = pointsPrevious(indexPairs(:, 2), :);

    % accumulate transformations
    tforms(n) = estgeotform2d(matchedPoints, matchedPointsPrev, ...
        'projective', 'MaxNumTrials', maxFastrTrials, 'Confidence', fastrConf, 'MaxDistance', fastrInlier);
        

    % make transformation relative to the first image
    tforms(n).A = tforms(n - 1).A * tforms(n).A;

    % Update features for the next iteration
    featuresPrevious = features;
    pointsPrevious = pointsR;
end    

% initialize x and y limits on transformations
xlim = zeros(numImages, 2);
ylim = zeros(numImages, 2);

% limits
for i = 1:numel(tforms)           
    [xlim(i,:),ylim(i,:)] = outputLimits(tforms(i),[1 imageSize(i,2)],[1 imageSize(i,1)]);    
end

% initialize the panorama
for i = 1:numel(tforms)           
    [xlim(i,:),ylim(i,:)] = outputLimits(tforms(i),[1 imageSize(i,2)],[1 imageSize(i,1)]);
end
maxImageSize = max(imageSize);

% get the output limits
xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);
yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);

width  = round(xMax - xMin);
height = round(yMax - yMin);

% blank canvas for panorama
S4panorama = zeros([height width 3],"like",img);

% size of panorama
panoramaView = imref2d([height width],[xMin xMax],[yMin yMax]);

% make the panorama!!!!!!!!!!!!!!!!!!
for i = 1:numImages
    img = S4_imgSet{i};   
    warpedImage = imwarp(img,tforms(i),'OutputView',panoramaView);                 
    mask = imwarp(true(size(img,1),size(img,2)),tforms(i),'OutputView',panoramaView);
    S4panorama = imblend(warpedImage,S4panorama,mask,foregroundopacity=1);
end

imshow(S4panorama)
imwrite(S4panorama, 'S2-largepanorama.png')