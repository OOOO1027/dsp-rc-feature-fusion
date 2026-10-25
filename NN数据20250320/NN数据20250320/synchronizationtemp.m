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


data1=importdata('./tb_rxdata/data1.txt');
data1=data1/sqrt(mean(abs(data1).^2));

data2=importdata('./tb_rxdata/data2.txt');
data2=data2/sqrt(mean(abs(data2).^2));
% dataout= importdata('./txdata/data36QAM.txt');
dataout=importdata('./txdata/RF0.105_data36QAM.txt');
%% Tx qamdata
load v1.txt
load v2.txt
NLTCP_qamdata = v1+1i*v2;
avp1=sqrt(mean(abs(NLTCP_qamdata).^2));
%% synchronization
[hicorrI,lagsiI1]=xcorr(real(data1),real(dataout));
[value,offsetindex1]=max(abs(hicorrI));
h=1;
if lagsiI1(offsetindex1)+length(dataout)+h-1>length(data1)
    hicorrI_new = hicorrI;
    hicorrI_new(offsetindex1) = 0;
    [value,offsetindex1]=max(abs(hicorrI_new));
end
figure;plot(lagsiI1,abs(hicorrI));

[hicorrI,lagsiI2]=xcorr(real(data2),real(dataout));
[value,offsetindex2]=max(abs(hicorrI));
h=1;
if lagsiI2(offsetindex2)+length(dataout)+h-1>length(data2)
    hicorrI_new = hicorrI;
    hicorrI_new(offsetindex2) = 0;
    [value,offsetindex2]=max(abs(hicorrI_new));
end
figure;plot(lagsiI2,abs(hicorrI));

lag1 = lagsiI1(offsetindex1);
yu1 = rem(lag1,length(dataout));
shang1 = (lag1-yu1)/length(dataout);
lag2 = lagsiI2(offsetindex2);
yu2 = rem(lag2,length(dataout));
shang2 = (lag2-yu2)/length(dataout);

for n=1:19

datarx1=data1(lagsiI1(offsetindex1)-(shang1-n+1)*length(dataout)+h:lagsiI1(offsetindex1)-(shang1-n)*length(dataout)+h-1);
datarx2=data2(lagsiI2(offsetindex2)-(shang2-n+1)*length(dataout)+h:lagsiI2(offsetindex2)-(shang2-n)*length(dataout)+h-1);
datarx1=datarx1/sqrt(mean(abs(datarx1).^2));
datarx2=datarx2/sqrt(mean(abs(datarx2).^2));
name = ['.\rxdata\rx1_', num2str(n)];
eval(['save ' name '.txt datarx1 -ASCII']);
name = ['.\rxdata\rx2_', num2str(n)];
eval(['save ' name '.txt datarx2 -ASCII']);
end

