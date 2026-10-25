close all;
clear;

index = 41;
awg_SampleRate = 3e9;

%%
constellation = 'APSK';
QAMorder=64;
numofsymbols=51200;
upsampleno=4;
alpha=0.205;
subcar=1/2;
startFreq=1/100;

%%
nametx = ['data',num2str(QAMorder),constellation,'.txt'];
dataout= importdata(nametx);
namerx = ['seedOSC_oldcap',num2str(QAMorder),constellation,num2str(index),'.txt'];
datarx= importdata(namerx);

%%
dataout = reshape(dataout,1,[]);
dataout = dataout/sqrt(mean(abs(dataout).^2));
datarx  = reshape(datarx,1,[]);
datarx  = datarx/sqrt(mean(abs(datarx).^2));
signal_out = datarx;
signal_in  = dataout;
figure;
plot(signal_in,signal_out,'b.');     
xfre=((0:1:length(dataout)-1)-length(dataout)/2)*awg_SampleRate/1e6/length(dataout); 
DataCap_f=fftshift(fft(dataout));
figure;plot(xfre,20*log10(abs(DataCap_f)),'b-')
title('Frequency Response');xlabel('Bandwidth(MHz)');ylabel('Amplitude(dB)')
%% waveform LMS volterra
% for taps_LMS=11:2:51
% fprintf("taps: %d: ", taps_LMS);
% for u_LMS = 0.01:0.01:0.07
% fprintf("u: %f: ", u_LMS);
% for taps_volterra=1:2:13
% fprintf("taps: %d: ", taps_volterra);
% for u_volterra = 0.0001:0.0001:0.001
% fprintf("u: %f: ", u_volterra);
taps_LMS=25;
u_LMS=0.01;
taps_volterra=7;
u_volterra=0.0005;
numofTs=8000;
[data_AfRLS,~,~,~]=LMS_volterra_1DownS_Testnan(taps_LMS,u_LMS,taps_volterra,u_volterra,numofTs,datarx,dataout);
nameLMS = ['afterLMS', num2str(index),'.txt'];
save(nameLMS, 'data_AfRLS',  '-ASCII');
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
%% Matched Filtering
taps = 35; 
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
%% LMS Post Equalization
namepre = 'decimal_';
namepost='_APSK.txt';
name2=[namepre,num2str(QAMorder),namepost];
DECdata=importdata(name2);
qamdata1 = GS_CCSDSmodulation_cons(length(DECdata),DECdata,QAMorder,constellation);
% for taps_LMS=11:4:71
% fprintf("taps: %d: ", taps_LMS);
% for u_LMS = 0.01:0.01:0.2
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
figure;plot(recoverdata(taps_LMS:end-taps_LMS),'.');
gscatter(real(recoverdata),imag(recoverdata),DECdata);hold on
%%
rxdata_dec=GS_CCSDSdemodulation_cons(length(recoverdata),recoverdata,QAMorder,constellation);
error=recoverdata(rxdata_dec~=DECdata);
plot(error,'k.','markersize',8)
[symnum,ser]=symerr(rxdata_dec(taps_LMS:end-taps_LMS),DECdata(taps_LMS:end-taps_LMS));
[bitnum,ber]=biterr(rxdata_dec(taps_LMS:end-taps_LMS),DECdata(taps_LMS:end-taps_LMS));
fprintf('SER  BER\t%.4e    \t%.4e\n', ser, ber);

% end
