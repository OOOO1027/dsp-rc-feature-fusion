close all;
clear;

%%
constellation = 'APSK';
index = 42;
name = ['Rxdataw2w_afterNN', num2str(index), '.txt'];
datarx= importdata(name);
dataout= importdata('data64APSK.txt');

%% 
numofsymbols=51200;
QAMorder=64;
upsampleno=4;
alpha=0.205;
subcar=1/2;
startFreq=1/100;

a=0:QAMorder-1;
b=GS_CCSDSmodulation_cons(length(a),a,QAMorder,constellation);
avp=sqrt(mean(abs(b).^2));

%%
dataout = reshape(dataout,1,[]);
dataout=dataout/sqrt(mean(abs(dataout).^2));
signal_in=dataout/max(abs(dataout));    
datarx = reshape(datarx,1,[]);
datarx=datarx/sqrt(mean(abs(datarx).^2));
signal_out=datarx/max(abs(datarx));          
figure;
plot(signal_in,signal_out,'b.');    
x1=-1:0.001:1;
y1=-1:0.001:1;
hold on; plot(x1,y1,'r.');hold on;title('AM-AM');
mse = mean((signal_in - signal_out).^2);

awg_SampleRate = 3e9;
xfre=((0:1:length(dataout)-1)-length(dataout)/2)*awg_SampleRate/1e6/length(dataout);
figure;
plot(xfre,20*log10(abs(fftshift(fft(signal_in)))),'b-')
title('Frequency Response at Tx');xlabel('Bandwidth(MHz)');ylabel('Amplitude(dB)');grid on;figure;
plot(xfre,20*log10(abs(fftshift(fft(signal_out)))),'b-')
title('Frequency Response after NN');xlabel('Bandwidth(MHz)');ylabel('Amplitude(dB)');grid on;
%% Generate shaping filter
% disp('SRC filter generation...')
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

%% waveform LMS volterra   由NN的均衡输出代替
%  disp('Begin LMS Volterra...');
%  for taps_LMS=11:4:55
% fprintf("taps: %d: ", taps_LMS);
% for u_LMS = 0.001:0.001:0.02
% fprintf("u: %f: ", u_LMS);
%  for taps_volterra=1:2:13
% fprintf("taps: %d: ", taps_volterra);
% for u_volterra = 0.0001:0.0001:0.001
% fprintf("u: %f: ", u_volterra);

% taps_LMS=35;
% u_LMS=0.01;
% taps_volterra=7;
% u_volterra=0.0005;
% numofTs=8000;
% [data_AfRLS,data_Ts,e,W]=LMS_1DownS_Testnan(taps_LMS,u_LMS,numofTs,datarx,dataout);
data_AfRLS = datarx;

%% Matched Filtering
taps = 35; %  filter length (should be odd number !!) 
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
MatchDefilter_data = MatchDefilter_data./sqrt(mean(abs(MatchDefilter_data).^2))*avp;

%%
namepre = 'decimal_';
namepost='_APSK.txt';
name2=[namepre,num2str(QAMorder),namepost];
DECdata=importdata(name2);
qamdata1 = GS_CCSDSmodulation_cons(length(DECdata),DECdata,QAMorder,constellation);

%% 2nd LMS
%  for taps_LMS=11:2:51
%     fprintf("taps: %d: ", taps_LMS);
%     for u_LMS = 0.0001:0.0001:0.002
%     fprintf("u: %f: ", u_LMS);
    taps_LMS = 25;
    u_LMS = 0.01;
    numofTs=8000;
      [data_AfRLS,data_Ts,e,W]=LMS_1DownS_Testnan(taps_LMS,u_LMS,numofTs,MatchDefilter_data,qamdata1);
    recoverdata=data_AfRLS;
    % recoverdata=MatchDefilter_data;
    recoverdata=recoverdata./sqrt(mean(abs(recoverdata).^2))*avp;
    recoverdata=reshape(recoverdata,1,[]);

    origindata=qamdata1;
    DECdata=reshape(DECdata,1,[]);
    recoverdata=reshape(recoverdata,1,[]);
    figure;plot(recoverdata(taps_LMS:end-taps_LMS),'.');
    gscatter(real(recoverdata),imag(recoverdata),DECdata);hold on

    %%
    rxdata_dec=GS_CCSDSdemodulation_cons(length(recoverdata),recoverdata,QAMorder,constellation);
    error=recoverdata(rxdata_dec~=DECdata);
    plot(error,'k*')
    [symnum,ser]=symerr(rxdata_dec(taps_LMS:end-taps_LMS),DECdata(taps_LMS:end-taps_LMS));
    [bitnum,ber]=biterr(rxdata_dec(taps_LMS:end-taps_LMS),DECdata(taps_LMS:end-taps_LMS));
    fprintf('SER  BER\t%.4e    \t%.4e\n', ser, ber);
%  end

%% plot 
trainloss = importdata('train_loss.txt');
valloss = importdata('val_loss.txt');
epoch = importdata('epoch.txt');
figure;
plot(epoch,trainloss,'b','linewidth',1.5);
hold on;
plot(epoch,valloss,'r','linewidth',1.5);
legend('Train loss','Test loss');
xlabel('Epoch');ylabel('MSE');