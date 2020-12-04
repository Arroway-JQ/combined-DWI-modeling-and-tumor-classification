function [params,paramsmap] = getSMmap(mainDir,patientname,x,y,z,A,normdecay,b)
% A---the size of orginal 3D images
% x,y,z---the remembered coordinates of original 3D images
% b--- b values;
% mainDir and patientname are the dictionary to save IVIM
% normdecay&onormdecay---onormdecay is the original decay data while
% normdecay may partially approximated by small value epsilong.
ai = size(normdecay);
m = ai(1);
%ful fit
ADCS = zeros(1,m);
theta = zeros(1,m);
r2 = zeros(1,m);
parfor i = 1:m
    yysm = log(normdecay(i,:));
    [sm,P] = polyfit(b,yysm,2);
    %mdl = polyfitn(b,yysm,2);
    %sm = mdl.Coefficients;
    %r2(i) = mdl.R2;
    r2(i) = 1-(P.normr/norm(yysm-mean(yysm)))^2;
    ADCS(i) = -sm(2);
    theta(i) = real(sqrt(2*sm(1)));
end
params = {ADCS;theta;r2};

ADCSmap = zeros(A(1),A(2),A(3));
thetamap = zeros(A(1),A(2),A(3));
for i = 1:m
    ADCSmap(x(i),y(i),z(i)) = ADCS(i);
    thetamap(x(i),y(i),z(i)) = theta(i);
end
paramsmap = {ADCSmap;thetamap};
%save([mainDir 'SM/' 'sm' patientname],'params','paramsmap');
save([mainDir 'SM_nature/' 'sm' patientname],'params','paramsmap');
%save([mainDir 'WholeSM/' 'sm' patientname],'params','paramsmap');
fprintf('SM done');