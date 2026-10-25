%% pre processing for wave2sym NN 
clear;
close all;

index = 42;

DECdata = importdata('decimal_64_APSK.txt');
name = ['afterLMS',num2str(index), '.txt'];
RXdata = importdata(name);
TXdata = importdata('data64APSK.txt');

taps = 25;           % odd number!!
upsampleno = 4;
APSKorder = 64;
constellation = 'APSK';
datalen = length(DECdata);
numofsymbols = datalen*upsampleno;
a=0:APSKorder-1;
b=GS_CCSDSmodulation_cons(length(a),a,APSKorder,constellation);
avp=sqrt(mean(abs(b).^2));


%% constellation label
moddata = GS_CCSDSmodulation_cons(length(DECdata),DECdata,APSKorder,constellation);
moddatar = real(moddata);
moddatai = imag(moddata);
label = zeros(datalen, 2);
label(:, 1) = moddatar;
label(:, 2) = moddatai;

nameTX = 'Txdata_NN.txt';
eval(['save ' nameTX '  label  -ASCII']);


rxdata = reshape(RXdata,upsampleno,datalen);
rxdata_pad = zeros(upsampleno,datalen+taps-1);
rxdata_pad(:,(taps-1)/2+1 : (taps-1)/2+datalen) = rxdata;

rx = zeros(taps*upsampleno,datalen);
for i = (taps-1)/2+1:datalen+(taps-1)/2
    temp = rxdata_pad(:,i-((taps-1)/2) : i+(taps-1)/2) ;
    temp = reshape(temp,1,taps*upsampleno);
    rx(:,i-((taps-1)/2)) = temp;
end

rx = rx.';
nameRX = 'Rxdata_NN.txt';
eval(['save ' nameRX '  rx  -ASCII']);

