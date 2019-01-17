%INTELLIGENT TOOLS FOR ENGINEERING APPLICATIONS
%ENGI 5173 WA
%COURSE INSTRUCTOR: Dr. Wilson Wang
%Name: Venkat Sai Manoj Cheerla
%s.id: 0863427

clc;
close all; 
clear all;
%IHere i am building an anfis same as figure. 12.10(3 inputs, 2 MF for each input, 1 output) in book 2
a        = 0.2;     % value for a in eq (1)
b        = 0.1;     % value for b in eq (1)
tau      = 17;		% delay constant in eq (1)
x0       = 1.2;		% initial condition: x(t=0)=x0
deltat   = 0.1;	    % time step size (which coincides with the integration step)
sample_n = 12000;	% total no. of samples, excluding the given initial condition
interval = 1;	    % output is printed at every 'interval' time steps

time = 0;
index = 1;
history_length = floor(tau/deltat);
x_history = zeros(history_length, 1); % here we assume x(t)=0 for -tau <= t < 0
x_t = x0;

X = zeros(sample_n+1, 1); % vector of all generated x samples
T = zeros(sample_n+1, 1); % vector of time samples

for i = 1:sample_n+1,
    X(i) = x_t;
    if (mod(i-1, interval) == 0),
         disp(sprintf('%4d %f', (i-1)/interval, x_t));
    end
    if tau == 0,
        x_t_minus_tau = 0.0;
    else
        x_t_minus_tau = x_history(index);
    end

    x_t_plus_deltat = mackeyglass_rk4(x_t, x_t_minus_tau, deltat, a, b);

    if (tau ~= 0),
        x_history(index) = x_t_plus_deltat;
        index = mod(index, history_length)+1;
    end
    time = time + deltat;
    T(i) = time;
    x_t = x_t_plus_deltat;
end


figure
plot(T, X);
set(gca,'xlim',[0, T(end)]);
xlabel('t');
ylabel('x(t)');
title(sprintf('A Mackey-Glass time serie (tau=%d)', tau));

trn_data = zeros(500, 4);
chk_data = zeros(500, 4);

% prepare training data
trn_data(:, 1) = X(101:600);
trn_data(:, 2) = X(109:608);
trn_data(:, 3) = X(117:616);
trn_data(:, 4) = X(125:624);

% prepare checking data
chk_data(:, 1) = X(601:1100);
chk_data(:, 2) = X(609:1108);
chk_data(:, 3) = X(617:1116);
chk_data(:, 4) = X(625:1124);
h=trn_data(:,1:3);
d=trn_data(:,4);
x1=chk_data(:,1:3);
d1=chk_data(:,4);
npt=size(x1,2);
%Defining all the membership functions in layer 1
in1=[1:21];
in2=[-10:10];
in3=[5:25];
%3 membership functions for each input
[mf11]=myfun(in1,1,8,12);
[mf12]=myfun(in1,8,12,21);
[mf21]=myfun(in2,-10,-5,-1);
[mf22]=myfun(in2,-5,-1,10);
[mf31]=myfun(in3,5,11,17);
[mf32]=myfun(in3,11,17,25);
%Combining the membership functions of each input to plot
commf1 = [mf11;mf12];
commf2 = [mf21;mf22];
commf3 = [mf31;mf32];

%Plotting the membership functions
figure(2)
subplot(1,3,1);
plot(in1,[commf1])
title('Starting MF of input 1');
xlabel('First input')
ylabel('Membership functions mf11 mf12')
subplot(1,3,2);
plot(in2,[commf2])
title('Starting MF of input 2');
xlabel('Second input')
ylabel('Membership functions mf21 mf22')
subplot(1,3,3);
plot(in3,[commf3])
title('Starting MF of input 3');
xlabel('Third input')
ylabel('Membership functions mf31 mf32')

%Now we need to calculate the firing strengths in the next layer

w1=mf11.*mf21.*mf31;
w2=mf11.*mf21.*mf32;
w3=mf11.*mf22.*mf31;
w4=mf11.*mf22.*mf32;
w5=mf12.*mf21.*mf31;
w6=mf12.*mf21.*mf32;
w7=mf12.*mf22.*mf31;
w8=mf12.*mf22.*mf32;

%Now we need to normaalize the firing strength in the next layer
for m=1:npt
    if (w1(:,m)==0 && w2(:,m)==0 && w3(:,m)==0 && w4(:,m)==0 && w5(:,m)==0 && w6(:,m)==0 && w7(:,m)==0 && w8(:,m)==0)
        norw1(:,m)=0;norw2(:,m)=0;
        norw3(:,m)=0;norw4(:,m)=0;
        norw5(:,m)=0;norw6(:,m)=0;
        norw7(:,m)=0;norw8(:,m)=0;
        
    else
        wt(:,m)= w1(:,m)+w2(:,m)+w3(:,m)+w4(:,m)+w5(:,m)+w6(:,m)+w7(:,m)+w8(:,m);
        norw1(:,m)=w1(:,m)/wt(:,m);
        norw2(:,m)=w2(:,m)/wt(:,m);
        norw3(:,m)=w3(:,m)/wt(:,m);
        norw4(:,m)=w4(:,m)/wt(:,m);
        norw5(:,m)=w5(:,m)/wt(:,m);
        norw6(:,m)=w6(:,m)/wt(:,m);
        norw7(:,m)=w7(:,m)/wt(:,m);
        norw8(:,m)=w8(:,m)/wt(:,m);
        
    end
end
%In the below equation we have considered consequent parameters to be 1
X_I=[norw1.*h+norw2.*h+norw3.*h+norw4.*h+norw5.*h+norw6.*h+norw7.*h+norw8.*h+norw1+norw2+norw3+norw4+norw5+norw6+norw7+norw8];

figure(3)
plot(h,X_I);
xlabel('IN');
ylabel('OUT');
title('Our ANFISystem Final output');

%TRAINING 

erthreshold = 0.00001;	
lr = 0.1;		% Learning rate
alpha = 0.9;		% Momentum term
epochs = 500;	% Max. training epochs
noin = 3;		% Number of inputs
nohi = 3;	% Number of hidden units
noop = 1;		% Number of outputs

[row, col] = size(trn_data);
if noin + noop ~= col,
	error('Given data mismatches given I/O numbers!');
end
X0 = trn_data(:, 1:noin);
T = trn_data(:, noin+1:noin+noop);

%  Initialize parameters
% CENTER(i, j) is the j-th component of i-th center
input_range = max(X0) - min(X0);
C = rand(nohi, noin).*(ones(nohi, 1)*input_range) + ...
	ones(nohi, 1)*min(X0);
if noin == 1,
	C = linspace(min(X0), max(X0), nohi)';
end

% variance for i-th center
Var = 0.02*ones(nohi, 1); 
Var = 1/(2*length(C)-2)/sqrt(2*log(2))*ones(nohi, 1); % for SISO
Var = 1/(2*size(C, 1)^(1/noin)-2)/sqrt(2*log(2))*ones(nohi, 1);

initialW = .5;		% initial weights
W = initialW*2*(rand(nohi,noop) - 0.5);	
W = zeros(nohi, noop);

RMSE = zeros(epochs, 1);	% Root mean squared error
distance = zeros(row, nohi);

for i = 1:epochs,
	% Find distance matrix: dist(i,j) = distance from data i to center j
	distance = vecdist(X0, C);

	% feed Forward 
	i1 = exp(-(distance.^2)*diag(1./(2*Var.^2)));	% hidden layer
	i2 = i1*W;					% output layer
	difference = T - i2;	% error
	RMSE(i) = sqrt(sum(sum(difference.^2))/length(difference(:)));
	fprintf('Epoch %.0f:  RMSE = %.10g\n',i, RMSE(i));
	if RMSE(i) < erthreshold, break; end

	% Back Propagation for output layer
	dE_dX2 = -2*(T - i2);	% dE/dX1
	dE_dW = i1'*dE_dX2;

	% BP for hidden layer (radial basis functions)
	dE_di1 = dE_dX2*W';			% dE/dX1
	di1_dvar = i1.*(distance.^2*diag(Var.^(-3)));

	dE_dvar = sum(dE_di1.*di1_dvar)';
	dE_dC = diag(Var.^(-2))*((dE_di1.*i1)'*X0-diag(sum(dE_di1.*i1))*C);

	% Simple steepest descent
	dW = -lr*dE_dW;
	dvar = -lr*dE_dvar;
	dC = -lr*dE_dC;
	W = W + dW;
	Var = Var + dvar;
	C = C + dC;
end

if i < epochs,
	fprintf('Error goal reached after %g epochs.\n', i);
else
	fprintf('Max. no. of epochs (%g) reached.\n', epochs);
end
RMSE(i+1:epochs) = []; 
fprintf('Final RMSE: %.10g\n', RMSE(i));
figure(); 
plot(1:i, RMSE, '-', 1:i, RMSE, '>');
xlabel('Number of Epochs'); 
ylabel('RMSE-Root mean squared error');
title('Epoch vs RMSE');

