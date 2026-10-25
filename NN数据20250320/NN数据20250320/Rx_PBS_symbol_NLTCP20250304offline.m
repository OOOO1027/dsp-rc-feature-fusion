clear all
close all
%% this program is PBS diversity, symbol combination
datano=51200;%datano
NLTCP_QAMorder=36;
DB_QAMorder=16;
DB_QAMorder_2=49;

BitsPerSymbol=4;
PAMorder=4;
upsampleno=4;
alpha=0.105;
subcar=1/2;
startFreq=1/100;
%% Tx qamdata
load v1.txt
load v2.txt
NLTCP_qamdata = v1+1i*v2;
avp1=sqrt(mean(abs(NLTCP_qamdata).^2));
%% load rx data
for n=1:19
% n=1;
name1 = ['./rxdata/rx1_',num2str(n),'.txt'];
name2 = ['./rxdata/rx2_',num2str(n),'.txt'];
datarx1=importdata(name1);
datarx2=importdata(name2);
datarx1=datarx1/sqrt(mean(abs(datarx1).^2));
datarx2=datarx2/sqrt(mean(abs(datarx2).^2));

% dataout= importdata('./txdata/data36QAM.txt');
dataout=importdata('./txdata/RF0.105_data36QAM.txt');

%% CH1 waveform LMS volterra
 % disp('Begin ch1 LMS Volterra...');
%  for taps_LMS=3:2:55
% fprintf("taps: %d: ", taps_LMS);
% for u_LMS = 0.01:0.001:0.03
% fprintf("u: %f: ", u_LMS);
%  for taps_volterra=15:2:33
% fprintf("taps: %d: ", taps_volterra);
% for u_volterra = 0.00001:0.00001:0.0001
% fprintf("u: %f: ", u_volterra);
taps_LMS=33;
u_LMS=0.022;
taps_volterra=17;
u_volterra=0.00008;
numofTs1=10000;
[data_AfLMS1,data_Ts,e,W]=LMS_volterra_1DownS_Testnan(taps_LMS,u_LMS,taps_volterra,u_volterra,numofTs1,datarx1,dataout);

%% CH1 CAP demodulation
datarx1 = CAPdemodulation(data_AfLMS1,upsampleno,datano);
% plot(datarx1,'.')
%% CH1 symbol LMS 
%  for NLTCP_taps_LMS2=103:2:131
% fprintf("taps: %d: ", NLTCP_taps_LMS2);
% for NLTCP_u_LMS2 = 0.002:0.0001:0.003
% fprintf("u: %f: ", NLTCP_u_LMS2);
NLTCP_taps_LMS2=115; 
NLTCP_u_LMS2=0.0021;  
numofTs1=10000;

[data_AfRLS1,~,e,W]=LMS_1DownS_Testnan(NLTCP_taps_LMS2,NLTCP_u_LMS2,numofTs1,datarx1,NLTCP_qamdata);
%% CH1 NLTCP 解调
[dec_v1rx_PAM6,dec_v1rx_PAM4]=NLTCP_demodulation(datano,real(data_AfRLS1*avp1));
[dec_v2rx_PAM6,dec_v2rx_PAM4]=NLTCP_demodulation(datano,imag(data_AfRLS1*avp1));

% PAM6
load decimal_PAM4_1.txt;
load decimal_PAM4_2.txt;

% PAM4
[~,ber1]=biterr(dec_v1rx_PAM4(NLTCP_taps_LMS2:end-NLTCP_taps_LMS2),decimal_PAM4_1(NLTCP_taps_LMS2:end-NLTCP_taps_LMS2));
% disp(['Ber for PAM4 band1 is ',num2str(ber1)]);
[~,ber2]=biterr(dec_v2rx_PAM4(NLTCP_taps_LMS2:end-NLTCP_taps_LMS2),decimal_PAM4_2(NLTCP_taps_LMS2:end-NLTCP_taps_LMS2));
% disp(['Ber for PAM4 band2 is ',num2str(ber2)]);
ber_Rx1=(ber1+ber2)/2;
fprintf('BER for CH1 \t%.4e\n', ber_Rx1);
berall(n,1)=ber_Rx1;
%% CH3 waveform LMS volterra
% disp('Begin ch3 LMS Volterra...');
%  for taps_LMS=3:2:55
% fprintf("taps: %d: ", taps_LMS);
% for u_LMS = 0.01:0.001:0.03
% fprintf("u: %f: ", u_LMS);
%  for taps_volterra=15:2:33
% fprintf("taps: %d: ", taps_volterra);
% for u_volterra = 0.00001:0.00001:0.0001
% fprintf("u: %f: ", u_volterra);
taps_LMS=43;
u_LMS=0.013;
taps_volterra=13;
u_volterra=0.00005;
numofTs1=10000;
[data_AfLMS2,data_Ts,e,W]=LMS_volterra_1DownS_Testnan(taps_LMS,u_LMS,taps_volterra,u_volterra,numofTs1,datarx2,dataout);

%% CH3 CAP demodulation
datarx2 = CAPdemodulation(data_AfLMS2,upsampleno,datano);
% plot(datarx1,'.')
%% CH3 symbol LMS 
%  for NLTCP_taps_LMS2=103:2:131
% fprintf("taps: %d: ", NLTCP_taps_LMS2);
% for NLTCP_u_LMS2 = 0.002:0.0001:0.003
% fprintf("u: %f: ", NLTCP_u_LMS2);

NLTCP_taps_LMS2=115; 
NLTCP_u_LMS2=0.0011;  
numofTs1=10000;

[data_AfRLS2,~,e,W]=LMS_1DownS_Testnan(NLTCP_taps_LMS2,NLTCP_u_LMS2,numofTs1,datarx2,NLTCP_qamdata);
%% CH3 NLTCP 解调
[dec_v1rx_PAM6,dec_v1rx_PAM4]=NLTCP_demodulation(datano,real(data_AfRLS2*avp1));
[dec_v2rx_PAM6,dec_v2rx_PAM4]=NLTCP_demodulation(datano,imag(data_AfRLS2*avp1));

% PAM4
[~,ber1]=biterr(dec_v1rx_PAM4(NLTCP_taps_LMS2:end-NLTCP_taps_LMS2),decimal_PAM4_1(NLTCP_taps_LMS2:end-NLTCP_taps_LMS2));
% disp(['Ber for PAM4 band1 is ',num2str(ber1)]);
[~,ber2]=biterr(dec_v2rx_PAM4(NLTCP_taps_LMS2:end-NLTCP_taps_LMS2),decimal_PAM4_2(NLTCP_taps_LMS2:end-NLTCP_taps_LMS2));
% disp(['Ber for PAM4 band2 is ',num2str(ber2)]);
ber_Rx2=(ber1+ber2)/2;
fprintf('BER for CH2 \t%.4e\n', ber_Rx2);
berall(n,2)=ber_Rx2;
%% symbol combination
NLTCP_recoverdata=ber_Rx2/(ber_Rx1+ber_Rx2)*data_AfRLS1*avp1+ber_Rx1/(ber_Rx1+ber_Rx2)*data_AfRLS2*avp1;
%% 绘制普通星座图
% plot_hist3nan2025(NLTCP_recoverdata(NLTCP_taps_LMS2:end-NLTCP_taps_LMS2));
% %plot(NLTCP_recoverdata(NLTCP_taps_LMS2:end-NLTCP_taps_LMS2),'b.');
% title('NLTCP recoverdata');

%% NLTCP 解调
[dec_v1rx_PAM6,dec_v1rx_PAM4]=NLTCP_demodulation(datano,real(NLTCP_recoverdata));
[dec_v2rx_PAM6,dec_v2rx_PAM4]=NLTCP_demodulation(datano,imag(NLTCP_recoverdata));

% PAM6
load decimal_PAM4_1.txt;
load decimal_PAM4_2.txt;

% PAM4
[~,ber1]=biterr(dec_v1rx_PAM4(NLTCP_taps_LMS2:end-NLTCP_taps_LMS2),decimal_PAM4_1(NLTCP_taps_LMS2:end-NLTCP_taps_LMS2));
% disp(['Ber for PAM4 band1 is ',num2str(ber1)]);
[bitnum,ber2]=biterr(dec_v2rx_PAM4(NLTCP_taps_LMS2:end-NLTCP_taps_LMS2),decimal_PAM4_2(NLTCP_taps_LMS2:end-NLTCP_taps_LMS2));
% disp(['Ber for PAM4 band2 is ',num2str(ber2)]);
berall_PAM4=(ber1+ber2)/2;
fprintf('BER for PBS \t%.4e\n', berall_PAM4);
berall(n,3)=berall_PAM4;
end
save berall.txt berall -ascii;
plot(log10(berall))