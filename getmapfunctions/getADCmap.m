function [ADC,ADCmap] = getADCmap(mainDir,patientname,x,y,z,A,normdecay,b)
% x,y,z---remembered coordinates for 3D images;
% A----size of original 3D images;
% mainDir and patientname is the ditionary where to save ADC
% b---b values b=[0,50,100...]
ai = size(normdecay);
m = ai(1);
ADC = zeros(1,m);
r2 = zeros(1,m);
parfor i = 1:m
    sy = log(normdecay(i,:));
    [sl,P] = polyfit(b,sy,1);
    %mdl=polyfitn(b,sy,1);
    %sl = mdl.Coefficients;
    %r2(i) = mdl.R2;
    r2(i) = 1-(P.normr/norm(sy-mean(sy)))^2;
    ADC(i) = -sl(1);   
end
ADCmap = zeros(A(1),A(2),A(3));
ADC = [ADC;r2];
for i = 1:m
    ADCmap(x(i),y(i),z(i)) = ADC(1,i);
end
%save([mainDir 'ADC/' 'adc' patientname],'ADC','ADCmap');
save([mainDir 'ADC_nature/' 'adc' patientname],'ADC','ADCmap');
%save([mainDir 'WholeADC/' 'adc' patientname],'ADC','ADCmap');
fprintf('ADC done');