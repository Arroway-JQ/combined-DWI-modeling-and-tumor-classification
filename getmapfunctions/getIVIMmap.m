function [params,paramsmap] = getIVIMmap(mainDir,patientname,x,y,z,A,normdecay,onormdecay,b)
% A---the size of orginal 3D images
% x,y,z---the remembered coordinates of original 3D images
% b--- b values;
% mainDir and patientname are the dictionary to save IVIM
% normdecay&onormdecay---onormdecay is the original decay data while
% normdecay may partially approximated by small value epsilong.
ai = size(normdecay);
m = ai(1);
lb=length(b);
lb_mid=ceil(lb./2);
b_large = b(lb_mid+1:lb);
b_small = b(1:lb_mid);
% get initial value of f,Df,Ds
Ds_init=zeros(1,m); 
Df_init=zeros(1,m);
fslow = zeros(1,m);
ffast = zeros(1,m);
epsilong = 1*10^(-10);
% loop over each pixel
        % polyfit (1-f)exp(-bDs)
        % using large b data
parfor i = 1:m
    map = normdecay(i,lb_mid+1:lb);
    syh = log(map);
    ih = polyfit(b_large,syh,1);
    Ds_init(i) = -ih(1);
    fslow(i) = exp(ih(2));   
end
for i = 1:m
    map = normdecay(i,1:lb_mid);
    syl0(i,:) = map-fslow(i)*exp(-Ds_init(i)*b_small);
end
syl0(syl0 == 0) = epsilong;
parfor i = 1:m
    syl = real(log(syl0(i,:)));
    il = polyfit(b_small,syl,1);
    Df_init(i) =-il(1);
    ffast(i) = exp(il(2));
end
f_init = (fslow + (1-ffast))./2;
%%
%fprintf('finally fit all');
f=zeros(1,m);
Ds=zeros(1,m); 
Df=zeros(1,m);
RNew = zeros(1,m);
LB = [0,0,0];
UB = [1,1,1];
% finally fit f,Df,Ds by curvefit, using full data
        % x     :(f,Df,Ds)
        % xdata :b values
        % ydata :signals
parfor i = 1:m
    yi = onormdecay(i,:);       %   column vector of independent variable values
    X0 = [f_init(i),Df_init(i),Ds_init(i)];
    F = @(X,b) real((1-X(1))*exp(-b*X(2))+X(1)*exp(-b*X(3)));
    ff = optimoptions('lsqcurvefit','Algorithm','Levenberg-marquardt');    
    [X,resnorm] = lsqcurvefit(F,X0,b,yi,LB,UB,ff);
    sum_yi = sum(yi.^2);
    RNew(i) = 1-(resnorm/sum_yi)^(1/2);
    f(i) = X(1);
    Ds(i) = X(3);
    Df(i) = X(2);
end
params = {f;Df;Ds;RNew};
fmap = zeros(A(1),A(2),A(3));
Dsmap = zeros(A(1),A(2),A(3));
Dfmap = zeros(A(1),A(2),A(3));
for i = 1:m
    fmap(x(i),y(i),z(i)) = f(i);
    Dsmap(x(i),y(i),z(i)) = Ds(i);
    Dfmap(x(i),y(i),z(i)) = Df(i);
end
paramsmap = {fmap;Dfmap;Dsmap};
%save([mainDir 'IVIM/' 'ivim' patientname],'params','paramsmap');
save([mainDir 'IVIM_nature/' 'ivim' patientname],'params','paramsmap');
%save([mainDir 'WholeIVIM/' 'ivim' patientname],'params','paramsmap');
fprintf('IVIM done');