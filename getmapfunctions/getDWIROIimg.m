function [ROIimg] = getDWIROIimg(DWINII, DWIROINII, Flag_change,Flag_affine)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
    DWInii = double(DWINII.img);
    s_1 = size(DWInii);
    if Flag_change == 1
        
        imgdata = zeros(s_1(1),s_1(2),s_1(3),s_1(4));
        imgdata(:,:,:,1) = DWInii(:,:,:,s_1(4));
        imgdata(:,:,:,1) = DWInii(:,:,:,s_1(4));
        imgdata(:,:,:,2:s_1(4)) = DWInii(:,:,:,1:(s_1(4)-1));
    else
        imgdata = DWInii;
    end
    a = size(imgdata);
    label1 = double(DWIROINII.img);
    if Flag_affine == 1
       T1 = affine2d([-1 0 0; 0 1 0; a(1) 0 1]); 
       label = imwarp(label1,T1);
    else
       label = label1;
    end
    ROIimg = zeros(a(1),a(2),a(3),a(4));
    for k = 1:a(4)
        img_ROI = imgdata(:,:,:,k);
        img_ROI(label == 0) = 0;
        ROIimg(:,:,:,k) = img_ROI;
    end
end

