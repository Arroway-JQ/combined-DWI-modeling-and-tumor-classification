function [x,resnorm,residual,converge] = fitmod(funName,x0,xdata,ydata,lb,ub)

options = optimset('lsqcurvefit');
options = optimset(options, 'Algorithm',{'levenberg-marquardt',0.05}, ...
    'ScaleProblem','Jacobian','Display','off','TolFun',1.0e-5,'TolX',1.0e-6,'MaxIter',800);
% options =optimset('Algorithm','trust-region-reflective');



end
