% Training-only numerical cross-check; never changes the teaching sources.
here=fileparts(mfilename('fullpath'));base=fileparts(here);root=fileparts(base);
out=fullfile(base,'runs','phase2_509');
addpath(fullfile(root,'NN数据20250320','NN数据20250320'));
load(fullfile(out,'parity_input.mat'));
T=size(x,1);M=size(A,1);r=zeros(M,1);states=zeros(T,M);
for n=1:T
  sn=[x(n,:)';1];
  r=(1-gamma)*r+gamma*tanh(A*r+Win*sn);
  states(n,:)=r';
end
features=[states x ones(T,1)];
O=features(washout+1:end,:)';Y=target(washout+1:end,:)';
Wout=Y*O'/(O*O'+ridge*eye(size(O,1)));
pred=(Wout*features')';
% Independently construct interleaved complex-symbol windows in MATLAB.
taps=25;pad=[zeros((taps-1)/2,1);z;zeros((taps-1)/2,1)];x_window=zeros(T,2*taps);
for n=1:T
  chunk=pad(n:n+taps-1);x_window(n,:)=reshape([real(chunk)';imag(chunk)'],1,[])/input_scale;
end
cap=[];cap_status='not_run';
try
 cap=CAPdemodulation(wave.',4,length(wave)/4);cap_status='passed_execution';
catch err
 cap_status=err.message;
end
[q_i,d_i]=NLTCP_demodulation(T,real(symbols).');
[q_q,d_q]=NLTCP_demodulation(T,imag(symbols).');
bit_errors=[];bit_rate=[];biterr_status='not_run';
try
 [bit_errors,bit_rate]=biterr([0 1 2 3 4 5],[0 1 2 3 0 0]);biterr_status='passed_execution';
catch err
 biterr_status=err.message;
end
[lms_output,~,~,lms_W]=LMS_1DownS_Testnan(9,.001,256,z,symbols);
[vol_output,~,~,vol_W,vol_V]=LMS_volterra_1DownS_Testnan(9,.001,5,.00001,512,wave,txwave);
rng(100);rng_A=2*rand(80,80)-1;rng_A=rng_A*.9/abs(max(eig(rng_A)));rng_Win=2*rand(80,51)-1;
save(fullfile(out,'parity_matlab.mat'),'states','features','Wout','pred','x_window','cap','q_i','q_q','d_i','d_q','bit_errors','bit_rate','lms_output','lms_W','vol_output','vol_W','vol_V','rng_A','rng_Win','cap_status','biterr_status','-v7');
fprintf('MATLAB numerical cross-check completed on training-only fixture.\n');
