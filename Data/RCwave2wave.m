%% 波形级RC+符号级LMS
close all;
clear;

index =41;

dataTx = importdata('data64APSK.txt');
name = ['seedOSC_oldcap64APSK',num2str(index),'.txt'];
dataRx = importdata(name);

symbol = importdata('decimal_64_APSK.txt');
dataTx = reshape(dataTx, 1,[]);
dataRx = reshape(dataRx, 1,[]);
dataTx = dataTx/(sqrt(mean(abs(dataTx).^2)));
dataRx = -dataRx/(sqrt(mean(abs(dataRx).^2)));

%% Metaparameters   （节点数、衰减因子、谱半径）
taps = 25;  % odd number
outlen = 1;

M = 80;                       %储蓄池节点数
gamma = 0.7;                  %衰减

norm = 1e-5;                  %岭回归正则化参数
ratio = 0.1;                  %训练集占比
initiallen = ((taps-1)/2);    %空转长度
datalen = length(dataTx);

rng(100);
A = (rand(M,M)-0.5)*2;        %随机生成邻接矩阵A并根据谱半径sr缩放
sr = abs(max(eig(A))); 
A = A*0.5/sr;
Win = (rand(M,taps+1)-0.5)*2; %输入连接权重


dataRx_pad = zeros(1,datalen+taps-1);
dataRx_pad((taps-1)/2+1:(taps-1)/2+datalen) = dataRx;
datarx = zeros(datalen,taps);
for i = 1:datalen
   datarx(i,:) = dataRx_pad(i:i+taps-1); 
end
%% 空转、获取储蓄池状态变量、训练、回归
r = zeros(M,1);
sn = zeros(taps+1,1);
Ototal_train = zeros(M+taps+1,datalen*ratio-initiallen); 
Wout = zeros(outlen,M+taps+1);
Yd_train = dataTx(initiallen+1:ratio*datalen);
for n = 1:datalen*ratio
    sn(1:taps) = datarx(n,:);
    sn(taps+1) = 1;
    r = (1-gamma)*r + gamma * tanh( A * r + Win * sn );   % tanh/sigmoid/relu/linear
    o = [r;sn];
    if n > initiallen
        Ototal_train(:,n-initiallen) = o;
    end
end

I = eye(M+1+taps);
Wout = Yd_train * Ototal_train' / (Ototal_train * Ototal_train' + norm*I);
Yout_train = Wout * Ototal_train;
Yout_train = Yout_train / (sqrt(mean(abs(Yout_train).^2)));
mse_train = mean((Yd_train-Yout_train).^2);
nmse_train = (mean((Yd_train - Yout_train).^2)) / var(Yout_train); 

%% 验证
r = zeros(M,1);
sn = zeros(taps+1,1);
Ototal_test = zeros(M+1+taps,datalen-initiallen); 
Yd_test = dataTx(initiallen+1:datalen);
for n = 1:datalen
    sn(1:taps) = datarx(n, :);
    sn(taps+1) = 1;
    r = (1-gamma)*r + gamma*tanh( A * r + Win * sn );
    o = [r;sn];
    if n > initiallen
        Ototal_test(:,n-(taps-1)/2) = o;
    end
end

Yout_test = Wout * Ototal_test;
Yout_test = Yout_test / (sqrt(mean(abs(Yout_test).^2)));
nmse_test = (mean((Yd_test - Yout_test).^2)) / var(Yout_test); 
mse_test = mean((Yd_test-Yout_test).^2);

%%
% % %% save
% % r_out = Ototal_test.';
% % teacher_sym = Yd_test.';
% % 
% % namepre = 'BiRCw2w_Ototal';
% % namepost='.txt';
% % name=[namepre, num2str(index),namepost];
% % eval(['save ' name ' r_out  -ASCII']);
% % 
% % namepre = 'BiRC_teacher_wave';
% % namepost='.txt';
% % name=[namepre,num2str(index),namepost];
% % eval(['save ' name ' teacher_sym  -ASCII']);

%% 符号级LMS
Yout_test = reshape(Yout_test,1,[]);
datarx = [zeros(1,initiallen),Yout_test];
constellation = 'APSK';       
datatx = importdata('data64APSK.txt');        
% %%Save
% namepre = 'afterBiRCw2w';
% namepost='.txt';
% name=[namepre,num2str(index),namepost];
% save(name,'datarx','-ascii')
%% 
numofsymbols = 51200;
QAMorder  = 64;    
upsampleno = 4;
front = floor(initiallen/ 4)+1;
tail  = floor(initiallen/ 4)+1;
numofsymbols = numofsymbols-front-tail;
datatx = datatx(front*upsampleno+1:end-tail*upsampleno);
% name = ['afterRC',num2str(index),'.txt'];
% eval(['save ' name ' datarx  -ASCII']);

datarx = datarx(front*upsampleno+1:end-tail*upsampleno);

alpha=0.205;
subcar=1/2;
startFreq=1/100;

%%
datatx = reshape(datatx,1,[]);
datatx=datatx/sqrt(mean(abs(datatx).^2));
signal_in=datatx/max(abs(datatx));   
datarx = reshape(datarx,1,[]);
datarx=datarx/sqrt(mean(abs(datarx).^2));
signal_out=datarx/max(abs(datarx));          
x1=-1:0.001:1;
y1=-1:0.001:1;

%% Generate shaping filter
upsamplesymbol=numofsymbols*upsampleno;
t_sequence_norm=linspace(1,upsamplesymbol,upsamplesymbol)-1-upsamplesymbol/2;
gtr1    =cos(pi*t_sequence_norm*(1+alpha)*0.25)+sin(pi*t_sequence_norm*(1-alpha)*0.25)./(4*alpha*t_sequence_norm*0.25);
gtr2    =gtr1./(1-(4*alpha*t_sequence_norm*0.25).^2);
gtr     =gtr2*4*alpha/pi;
gtr(upsamplesymbol/2+1)=(1+alpha*(4/pi-1));
BW=1+alpha;
subcar1=BW*subcar+startFreq;
filter_I_path=reshape(gtr.*cos(2*pi*subcar1*t_sequence_norm*0.25),[],1);
filter_Q_path=reshape(gtr.*sin(2*pi*subcar1*t_sequence_norm*0.25),[],1);

data_AfRLS = datarx;
%% Matched Filtering
taps = 35;  %  filter length (should be odd number !!) 
offsetsample=0;
gtI = filter_I_path;
gtQ = filter_Q_path;
N = length(data_AfRLS);
filterI = gtI((1:taps)+N/2-(taps-1)/2);
filterQ = gtQ((1:taps)+N/2-(taps-1)/2);
DataCapI = conv(data_AfRLS,filterI,'same');
DataCapQ = conv(data_AfRLS,filterQ,'same');
DataCap=DataCapI+1i*DataCapQ;
MatchDefilter_data=downsample(DataCap,upsampleno,offsetsample);
%%
a=0:QAMorder-1;
b=GS_CCSDSmodulation_cons(length(a),a,QAMorder,constellation);
avp=sqrt(mean(abs(b).^2));

namepre = 'decimal_';
namepost='_APSK.txt';
name2=[namepre,num2str(QAMorder),namepost];
DECdata=importdata(name2);
DECdata = DECdata(front+1:end-tail);
qamdata1 = GS_CCSDSmodulation_cons(length(DECdata),DECdata,QAMorder,constellation);

%% 2nd LMS
% for taps_LMS = 11:2:51
% fprintf("taps: %d: ", taps_LMS);
% for u_LMS = 0.0001:0.0002:0.004
% fprintf("u: %f: ", u_LMS);
taps_LMS=25;
u_LMS=0.01;
numofTs=8000;
[data_AfRLS,data_Ts,e,W]=LMS_1DownS_Testnan(taps_LMS,u_LMS,numofTs,MatchDefilter_data,qamdata1);

recoverdata=data_AfRLS;
recoverdata=recoverdata./sqrt(mean(abs(recoverdata).^2))*avp;
recoverdata=reshape(recoverdata,1,[]);

origindata=qamdata1;
DECdata=reshape(DECdata,1,[]);
recoverdata=reshape(recoverdata,1,[]);
%%
figure;plot(recoverdata(taps_LMS:end-taps_LMS),'.');
gscatter(real(recoverdata),imag(recoverdata),DECdata);hold on

rxdata_dec=GS_CCSDSdemodulation_cons(length(recoverdata),recoverdata,QAMorder,constellation);
[symnum,ser]=symerr(rxdata_dec(taps_LMS:end-taps_LMS),DECdata(taps_LMS:end-taps_LMS));
[bitnum,ber]=biterr(rxdata_dec(taps_LMS:end-taps_LMS),DECdata(taps_LMS:end-taps_LMS));
error=recoverdata(rxdata_dec~=DECdata);
plot(error,'k.','markersize',8)
fprintf('BER\t%.7e\n', ber);
%  end
%%
figure;hold on;grid on;
set(gca,'FontSize',14,'FontName','Arial','Fontweight','bold','linewidth',1.5)
plot(Wout,'linewidth',1.5);
ylabel('Wout');
box(gca,'on');

% sym = [real(recoverdata); imag(recoverdata)];
% namepre = 'BiRCw2w_sym';
% namepost='.txt';
% name=[namepre,num2str(index),namepost];
% eval(['save ' name ' sym  -ASCII']);