function [x,resnorm,residual,converge] = fitmod(funName,x0,xdata,ydata,lb,ub)

options = optimset('lsqcurvefit');
options = optimset(options, 'Algorithm',{'levenberg-marquardt',0.05}, ...
    'ScaleProblem','Jacobian','Display','off','TolFun',1.0e-5,'TolX',1.0e-6,'MaxIter',800);
% options =optimset('Algorithm','trust-region-reflective');

exit = 0;
x0_fix = x0;
converge = 1;
num_fail = 0;
lb = []; ub = [];
% lb
% ub
while exit < 1
    try
        [x,resnorm,residual,exitflag] = lsqcurvefit(funName,x0_fix,xdata, ...
            ydata,lb,ub,options);
        exit = exitflag;
        x0_fix = x;
        if(exit == -3 || num_fail == 5)
            converge = false;
            break;
        elseif(exit < 1 && exit ~= -3)
            num_fail = num_fail + 1;
        else
        end
    catch err
        warning(err.identifier,'%s - %s:\n',err.identifier,err.message);
        disp(x0_fix);
        x = x0_fix .* 0;
        resnorm = 0;
        residual = zeros(size(xdata));
        converge = -1;
        exit = 10;
        return;
    end
end

end