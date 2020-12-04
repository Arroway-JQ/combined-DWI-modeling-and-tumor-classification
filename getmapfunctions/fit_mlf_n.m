function [mlf_coef,ChiSq,residuals,converge] = fit_mlf_n(xdata,ydata,D,A,B,weights)
% ChiSq 为 拟合度指标
if isempty(ydata)
    mlf_coef = [0 0 0];
    ChiSq = 0;
    residuals = zeros(1,numel(xdata));
    converge = -1;
else
    
    % 1) Normalize the data w.r.t. the initial value
    % data_norm = abs(ydata ./ S0);
    
    if isempty(weights)
        xdata = xdata(ydata ~= 0);
        ydata = ydata(ydata ~= 0);
    end
    
% initial values
    alpha_ini = A;
    beta_ini = B;

    x0 = [1 D alpha_ini beta_ini];
    lb = [1*0.1   0   0    0    ];
    ub = [1*10   1.0  1.5   3.0   ];
    if isempty(weights)
        [mlf_coef,resnorm,residuals,converge] = ...
            fitmod('mlf_model',x0,xdata,ydata,lb,ub);
    else
        [mlf_coef,resnorm,residuals,converge] = ...
            fitmod_weighted('mlf_model',x0,xdata,ydata,weights,lb,ub);
    end
    converge = double(converge);
    
    % 4) Calculate error (mean absolute percent error)
    sum_y = sum(ydata.^2);
    ChiSq = 1-(resnorm/sum_y)^(1/2);
    
end

end