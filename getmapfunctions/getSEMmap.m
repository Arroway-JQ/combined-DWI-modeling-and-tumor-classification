function [params,paramsmap,DDC_init] = getSEMmap(mainDir,patientname,x,y,z,A,normdecay,onormdecay,b)
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
epsilong = 1*10^(-10);
% get initial value 
DDC_init=zeros(1,m);
alpha_init=zeros(1,m); 
% parameters map
DDC=zeros(1,m);
alpha=zeros(1,m); 
RNew = zeros(1,m);
%fprintf('1st ployfit for DDC_init\n');
parfor i = 1:m
    ys = log(normdecay(i,1:lb_mid));
    ss = polyfit(b(1:lb_mid),ys,1);
    DDC_init(i) = -ss(1);
end
%for i = 1:m
%    if DDC_init(i)<0
%        DDC_init(i) = max(ADC(i),epsilong);
%    end
%end
%fprintf('2nd fit alpha_init')
b_large = b((lb_mid+1):lb);
parfor i = 1:m
    %map = normdecay(i,(lb_mid+1):lb);
    map = normdecay(i,(lb_mid+1):lb);
    AL0 = 0.9000;
    %SEMf = @(AL,b_large) real(exp(-(b_large*DDC_init(i)).^AL));
    SEMf = @(AL,b_large) real(exp(-(b_large*DDC_init(i)).^AL));
    ff = optimoptions('lsqcurvefit','Algorithm','Levenberg-marquardt');
    [AL,residual3,Jacobian3] = lsqcurvefit(SEMf,AL0,b_large,map,[],[],ff);
    alpha_init(i) = AL(1);
    alpha_init(i) = min(max(alpha_init(i),epsilong),1-epsilong);
end        
%%
%fprintf('full voxel fit');
parfor i = 1:m
    ysf = onormdecay(i,:);
    %ysf = log(onormdecay(i,:));
    C0 = [DDC_init(i),alpha_init(i)];
    SF = @(C,b) real(exp(-(b*C(1)).^C(2)));
    %SF = @(C,b) real(-(b*C(1)).^C(2));
    ff = optimoptions('lsqcurvefit','Algorithm','Levenberg-marquardt');
    [C,resnorm] = lsqcurvefit(SF,C0,b,ysf,[],[],ff);
    DDC(i) = C(1);
    alpha(i) = C(2);
    sum_ysf = sum(ysf.^2);
    RNew(i) = 1-(resnorm/sum_ysf)^(1/2);   
end
params = {DDC;alpha;RNew};
DDCmap = zeros(A(1),A(2),A(3));
alphamap = zeros(A(1),A(2),A(3));

for i = 1:m
    DDCmap(x(i),y(i),z(i)) = DDC(i);
    alphamap(x(i),y(i),z(i)) = alpha(i);
end
paramsmap = {DDCmap;alphamap};
%save([mainDir 'SEM/' 'sem' patientname],'params','paramsmap');
save([mainDir 'SEM_nature/' 'sem' patientname],'params','paramsmap');
%save([mainDir 'WholeSEM/' 'sem' patientname],'params','paramsmap');
fprintf('SEM done');
