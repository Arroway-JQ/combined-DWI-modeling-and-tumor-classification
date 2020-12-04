function [params,paramsmap] = getCTRWmap(mainDir,patientname,x,y,z,A,normdecay,onormdecay,b,DDC_init)
% A---the size of orginal 3D images
% x,y,z---the remembered coordinates of original 3D images
% b--- b values;
% mainDir and patientname are the dictionary to save IVIM
% normdecay&onormdecay---onormdecay is the original decay data while
% normdecay may partially approximated by small value epsilong.
%DDC_init get from SEM model
ai = size(normdecay);
m = ai(1);
S0 = 1;%because I have normalised signals in the data
Dm = DDC_init;

%%
alphac = zeros(1,m);
betac = zeros(1,m);
Dc  = zeros(1,m);
RNew = zeros(1,m);
parfor i = 1:m
    params = [1 Dm(i) 0.9 0.8];
    %S_true = mlf_model(params,b);
    S = onormdecay(i,:);
    [mlf_coef_fit,ChiSq,residuals,converge] = fit_mlf_n(b,S,Dm(i),params(3),params(4),[]);
    alphac(i) = mlf_coef_fit(3);
    betac(i)  = mlf_coef_fit(4);
    Dc(i) = mlf_coef_fit(2);
    RNew(i) = ChiSq;
end
params = {Dc;alphac;betac;RNew};
% params' map
Dcmap = zeros(A(1),A(2),A(3));
alphacmap = zeros(A(1),A(2),A(3));
betacmap = zeros(A(1),A(2),A(3));
for i = 1:m
    Dcmap(x(i),y(i),z(i)) = Dc(i);
    alphacmap(x(i),y(i),z(i)) = alphac(i);
    betacmap(x(i),y(i),z(i)) = betac(i);
end
paramsmap = {Dcmap;alphacmap;betacmap};
%save([mainDir 'CTRW/' 'ctrw' patientname],'params','paramsmap');
save([mainDir 'CTRW_nature/' 'ctrw' patientname],'params','paramsmap');
%save([mainDir 'WholeCTRW/' 'ctrw' patientname],'params','paramsmap');
fprintf('CTRW done');
