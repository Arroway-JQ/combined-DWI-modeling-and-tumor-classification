function [params,paramsmap] = getDKImap(mainDir,patientname,x,y,z,A,normdecay,b)
% A---the size of orginal 3D images
% x,y,z---the remembered coordinates of original 3D images
% b--- b values;
% mainDir and patientname are the dictionary to save IVIM
% normdecay&onormdecay---onormdecay is the original decay data while
% normdecay may partially approximated by small value epsilong.
ai = size(normdecay);
m = ai(1);
DK = zeros(1,m);
K = zeros(1,m);
r2 = zeros(1,m);
parfor i = 1:m
    yk = log(normdecay(i,:));
    [sk,P] = polyfit(b,yk,2);
    %mdl = polyfitn(b,yk,2);
    %sk = mdl.Coefficients;
    %r2(i) = mdl.R2
    r2(i) = 1-(P.normr/norm(yk-mean(yk)))^2;
    DK(i) = -sk(2);
    K(i) = (6*sk(1))/(-sk(2))^2;
end
params = {DK;K;r2};
DKmap = zeros(A(1),A(2),A(3));
Kmap = zeros(A(1),A(2),A(3));
for i = 1:m
    DKmap(x(i),y(i),z(i)) = DK(i);
    Kmap(x(i),y(i),z(i)) = K(i);
end
paramsmap = {DKmap;Kmap};
%save([mainDir 'DKI/' 'dki' patientname],'params','paramsmap');
save([mainDir 'DKI_nature/' 'dki' patientname],'params','paramsmap');
%save([mainDir 'WholeDKI/' 'dki' patientname],'params','paramsmap');
fprintf('DKI done');