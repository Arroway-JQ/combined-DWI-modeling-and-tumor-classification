function [params,paramsmap] = getFROC2map(mainDir,patientname,x,y,z,A,normdecay,onormdecay,b,DDC_init,alpha)
% A---the size of orginal 3D images
% x,y,z---the remembered coordinates of original 3D images
% b--- b values;
% mainDir and patientname are the dictionary to save IVIM
% normdecay&onormdecay---onormdecay is the original decay data while
% normdecay may partially approximated by small value epsilong.
% DDC_init and alpha get from SEM model
%Delta = 35.4;
%delta = 16.4;%breast的参数?

%Delta = 33.5;
%delta = 14.1;%renjibupto5000的参数

%Delta = 38.6;
%delta = 32.2; %pediatric的参数
Delta = 42.688;
delta = 29.404;%Adults的参数 renji bupto4500 the same machine so try the same params
%Delta = Delta-delta./3;
epsilong = 1*10^(-10);
betaf_init = alpha;
D_init = DDC_init;
ai = size(normdecay);
m = ai(1);
% fit mu_init and betaf_init
mu_init=zeros(1,m);
betaf_init2 = zeros(1,m);
Destimate = zeros(1,m);
bsmall=[0,20,50,80,150,200,300,500,800];
for i = 1:m
    sy = log(normdecay(i,1:9));
    [sl,P] = polyfit(bsmall,sy,1);
    %mdl=polyfitn(b,sy,1);
    %sl = mdl.Coefficients;
    %r2(i) = mdl.R2;
    Destimate(i) = -sl(1);   
end

parfor i = 1:m
    map = normdecay(i,:);
    A00 = [8.0000,betaf_init(i)];
    initf = @(AA,b) real(exp(-D_init(i).*AA(1).^(2.*(AA(2)-1)).*(b./(Delta-delta/3)).^AA(2).*(Delta-((2*AA(2)-1)/(2*AA(2)+1)).*delta)));
    ff = optimoptions('lsqcurvefit','Algorithm','Levenberg-marquardt');
    try
        [AA,residual4,Jacobian4] = lsqcurvefit(initf,A00,b,map,[],[],ff);
    catch
        mu_init(i) = 8.0000;
        betaf_init2(i) = betaf_init(i);
        continue;
    end
    mu_init(i) = AA(1);
    betaf_init2(i) = AA(2);
end

parfor i = 1:m
    map = normdecay(i,:);
    A00 = [mu_init(i),betaf_init2(i)];
    initf = @(AA,b) real(exp(-Destimate(i).*AA(1).^(2.*(AA(2)-1)).*(b./(Delta-delta/3)).^AA(2).*(Delta-((2*AA(2)-1)/(2*AA(2)+1)).*delta)));
    ff = optimoptions('lsqcurvefit','Algorithm','Levenberg-marquardt');
    try
        [AA,resnorm] = lsqcurvefit(initf,A00,b,map,[],[],ff);
    catch
        mu1(i) = mu_init(i);
        betaf1(i) = betaf_init2(i);
        RNew1(i) = 0;
        continue;
    end
    mu1(i) = AA(1);
    betaf1(i) = AA(2);
    sum_yf1 = sum(map.^2);
    RNew1(i) = 1-(resnorm/sum_yf1)^(1/2);
end
%%
% set beta and mu to fit D
%D = zeros(1,m);
%parfor i = 1:m
%    yd = log(normdecay(i,:));
%    bx = mu_init(i).^(2.*(betaf_init2(i)-1)).*(b./(Delta)).^betaf_init2(i).*(Delta+delta/3-((2*betaf_init2(i)-1)./(2.*betaf_init2(i)+1)).*delta);
%    sd = polyfit(bx,yd,1);
%    D0 = D_init;
%    initdf = @(DD,b) real(-DD(1).*mu_init(i).^(2.*(betaf_init2(i)-1)).*(b./(Delta)).^betaf_init2(i).*(Delta+delta/3-((2*betaf_init2(i)-1)./(2.*betaf_init2(i)+1)).*delta));
%    ff = optimoptions('lsqcurvefit','Algorithm','Levenberg-marquardt');
%    try
    %   [DD,residual4,Jacobian4] = lsqcurvefit(initdf,D0,b,yd,[],[],ff);
    %catch
    %    D(i) = D_init(i);
    %end
%    D(i) = real(-sd(1));
%end 
%%
% Set D to fit beta and mu
% full fit
D = zeros(1,m);
mu = zeros(1,m);
betaf = zeros(1,m);
RNew = zeros(1,m);
parfor i = 1:m
    yf = onormdecay(i,:);
    %yf = log(onormdecay(i,:));
    A0 = [D_init(i),mu_init(i),betaf_init2(i)];
    %resf = @(A,b) real(-A(1).*(A(2).^(2.*(A(3)-1))).*((b./(Delta-3/delta)).^A(3)).*(Delta-((2*A(3)-1)/(2*A(3)+1)).*delta));
    resf = @(A,b) real(exp(-A(1).*(A(2).^(2.*(A(3)-1))).*((b./(Delta-3/delta)).^A(3)).*(Delta-((2*A(3)-1)/(2*A(3)+1)).*delta)));
    ff = optimoptions('lsqcurvefit','Algorithm','Levenberg-marquardt');
    
    try
        [A,resnorm] = lsqcurvefit(resf,A0,b,yf,[],[],ff);
    catch
        D(i) = D_init(i);
        mu(i) = mu_init(i);
        betaf(i) = betaf_init2(i);
        RNew(i) = 0;
        continue;
    end
    D(i) = A(1);
    mu(i) = A(2);
    betaf(i) = A(3);
    sum_yf = sum(yf.^2);
    RNew(i) = 1-(resnorm/sum_yf)^(1/2);
end
params = {D;Destimate;mu;mu1;betaf;betaf1;RNew;RNew1};
%paramsinit = {D_init;mu_init;betaf_init2};
Dmap = zeros(A(1),A(2),A(3));
betafmap = zeros(A(1),A(2),A(3));
mumap = zeros(A(1),A(2),A(3));
%D_initmap = zeros(A(1),A(2),A(3));
%mu_initmap = zeros(A(1),A(2),A(3));
%betaf_init2map = zeros(A(1),A(2),A(3));
for i = 1:m
    Dmap(x(i),y(i),z(i)) = D(i);
    betafmap(x(i),y(i),z(i)) = betaf(i);
    mumap(x(i),y(i),z(i)) =mu(i);
    %D_initmap(x(i),y(i),z(i)) = D_init(i);
    %betaf_init2map(x(i),y(i),z(i)) = betaf_init2(i);
    %mu_initmap(x(i),y(i),z(i)) = mu_init(i);
end
paramsmap = {Dmap;mumap;betafmap};
%paramsinitmap = {D_initmap;mu_initmap;betaf_init2map};
%save([mainDir 'FROC/' 'froc' patientname],'params','paramsmap');
save([mainDir 'FROC_nature/' 'froc' patientname],'params','paramsmap');

fprintf('optimalFROC done');



