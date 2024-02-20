function grad_avere=gradient_average(f)
tf=double(f);
[grad_x,grad_y]=gradient(tf);
grad_avere=mean(mean(sqrt(grad_x .* grad_x + grad_y .* grad_y)));
