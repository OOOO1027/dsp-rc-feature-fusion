clear;
close all;

tx = importdata('data64APSK.txt');

rx = importdata('seedOSC_oldcap64APSK42.txt');

lms = importdata('afterLMS42.txt');
rc = importdata('afterBiRCw2w42.txt');
nn = importdata('Rxdataw2w_afterNN42.txt');

tx = reshape(tx,1,[]);
tx=tx/sqrt(mean(abs(tx).^2));
rx = reshape(rx,1,[]);
rx=rx/sqrt(mean(abs(rx).^2));
lms = reshape(lms,1,[]);
lms=lms/sqrt(mean(abs(lms).^2));
nn = reshape(nn,1,[]);
nn=nn/sqrt(mean(abs(nn).^2));

% figure;
% plot(tx,rx,'b.');     
% maxnum = 3 ;
% x1=-maxnum:maxnum;
% y1=x1;
% hold on; plot(x1,y1,'-r','linewidth',2);hold on;title('AM-AM');
% axis([0,maxnum,0,maxnum]) 
% figure;
% plot(tx,lms,'b.');     
% maxnum = 3;
% x1=-maxnum:maxnum;
% y1=x1;
% hold on; plot(x1,y1,'-r','linewidth',2);hold on;title('AM-AM');
% axis([0,maxnum,0,maxnum]) 
% figure;
% plot(tx,nn,'b.');     
% maxnum = 3 ;
% x1=-maxnum:maxnum;
% y1=x1;
% hold on; plot(x1,y1,'-r','linewidth',2);hold on;title('AM-AM');
% axis([0,maxnum,0,maxnum]) 

numofsymbols = 51180;
QAMorder  = 64;    
upsampleno = 4;
taps = 13;
initiallen = ((taps-1)/2);    %空转长度

front = floor(initiallen/ 4)+1;
tail  = floor(initiallen/ 4)+1;
numofsymbols = numofsymbols-front-tail;
datatx = tx(front*upsampleno+1:end-tail*upsampleno);
datarc = rc(front*upsampleno+1:end-tail*upsampleno);
datarc = reshape(datarc,1,[]);
datarc=datarc/sqrt(mean(abs(datarc).^2));

% figure;
% plot(datatx,datarc,'b.');     
% maxnum = 3 ;
% x1=-maxnum:maxnum;
% y1=x1;
% hold on; plot(x1,y1,'-r','linewidth',2);hold on;title('AM-AM');
% axis([0,maxnum,0,maxnum]) 

%%
awg_SampleRate = 3e9;
xfre=((0:1:length(tx)-1)-length(tx)/2)*awg_SampleRate/1e6/length(tx);

figure;
plot(xfre,20*log10(abs(fftshift(fft(tx)))),'color',[0.6,0.6,0.6],'linewidth',2.5);
xlabel('Bandwidth(MHz)');
ylabel('Amplitude(dB)');
grid on; hold on;

% % % % plot(xfre,20*log10(abs(fftshift(fft(tx_preEQU)))),'color',[82,255,87]/255,'linewidth',2.5)
% % % % legend('Tx','Tx-preEQU','location','southwest')

% plot(xfre,20*log10(abs(fftshift(fft(rx)))),'color',[0,0,255]/255,'linewidth',2.5);     %[97,179,255]/255
% legend('Tx','Rx','location','southwest')

% plot(xfre,20*log10(abs(fftshift(fft(lms)))),'color',[128,0,255]/255,'linewidth',2.5)
% legend('Tx','LMS-Volterra','location','southwest')

% plot(xfre,20*log10(abs(fftshift(fft(rc)))),'color',[255,20,30]/255,'linewidth',2.5)   %[0, 191, 64]/255,
% legend('Tx','BiRC','location','southwest')

% plot(xfre,20*log10(abs(fftshift(fft(nn)))),'color',[0,150,0]/255,'linewidth',2.5)
% legend('Tx','NN','location','southwest')

set(gca,'FontSize',30,'FontName','Arial','Fontweight','bold','linewidth',2.5)
set(gca,'Ytick',-40:40:80)

xlim([-45,max(xfre)]);
ylim([-40,80])