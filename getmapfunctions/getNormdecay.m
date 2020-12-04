function [Report] = getNormdecay(ROIimgDir, normdecayDir,mainDir,b,voxelindex)
%UNTITLED4 此处显示有关此函数的摘要
%   此处显示详细说明
%   voxelindex: (1,2) matrix;
    snormdecayall = [];
    svoxelnum = [];
    patientlist = dir(ROIimgDir);
    patientnum = size(patientlist);
    for i = 3:patientnum(1)
        cd (ROIimgDir)
        patientname = patientlist(i).name;
        DATA1 = load(patientname);
        OriROIdata = DATA1.ROIimg;
        A = size(OriROIdata);
        for j = 1:A(3)
            parfor k = 1:A(4)
                OriROIdata(:,:,j,k) = medfilt2(OriROIdata(:,:,j,k));  %removing noise
            end
        end
        b0map = OriROIdata(:,:,:,1);
        ovoxeldecay = reshape(OriROIdata,[A(1)*A(2)*A(3),A(4)]);
        C = b0map;
        for k = 1:(A(4)-1)
            C = and(C>0, OriROIdata(:,:,:,k+1)); % eliminating the zero signals
        end
        voxeldecay = ovoxeldecay(find(C>0),1:A(4));
        [x,y,z] = ind2sub(A,find(C>0));
        a = size(voxeldecay);
        m = a(1);
        normdecay = zeros(a);
        parfor n = 1:m
            normdecay(n,:) = voxeldecay(n,:)./voxeldecay(n,1);  %根据b0归一化
        end
        save([normdecayDir 'n' patientname],'normdecay','A','x','y','z');
        figure;
        plot(b,normdecay(voxelindex(1):voxelindex(2),:));
        snormdecayall = [snormdecayall;normdecay];
        svoxelnum = [svoxelnum;m];
        fprintf([patientname,':',num2str(m)]);
    end
    index = (linspace(1,patientnum(1)-2,patientnum(1)-2))';
    svoxelindex = [index,svoxelnum];
    save([mainDir,'sinputcurve_comp.mat'],'snormdecayall','svoxelindex');
   
    Report = 'Calculate normdecay done';
end

