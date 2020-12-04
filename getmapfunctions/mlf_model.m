function S = mlf_model(params,xdata)

S0 = 1;
D = params(2);
alpha = params(3);
beta = params(4);

%range of parameters in mlf diffusion model  : 0.5 < alpha,beta < 1 ; 
if (alpha>1 || beta >1 || alpha <0.5 || beta<0.5)
    S=nan;
    return
end

fi = 9;% more accurate with a larger fi number, but also slower.
S = S0 .* mlf(alpha,1,-(D.*xdata).^beta,fi); % E_a(-(bD)^beta)
end