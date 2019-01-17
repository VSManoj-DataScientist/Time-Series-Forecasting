%Defining membership functions
function [y] = myfun(x,a,b,c)
y = max(min((x-a)/(b-a),(c-x)/(c-b)),0);
end