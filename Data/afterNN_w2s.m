%% post processing and BER testing for wave2sym NN
clear;
close all;

index = 42;
DECdata = importdata('decimal_64_APSK.txt');
name = ['Rxdataw2s_afterNN',num2str(index), '.txt'];
RXdataNN = importdata(name); 

upsampleno = 4;
APSKorder = 64;
constellation='APSK';
datalen = length(DECdata);
numofsymbols = datalen*upsampleno;

a=0:APSKorder-1;
b=GS_CCSDSmodulation_cons(length(a),a,APSKorder,constellation);
avp=sqrt(mean(abs(b).^2));

moddata = GS_CCSDSmodulation_cons(length(DECdata),DECdata,APSKorder,constellation);

%% %for w2s using constellation label 
RXdata = RXdataNN(:,1) + 1j*RXdataNN(:,2);
RXdata = RXdata./sqrt(mean(abs(RXdata).^2)) * avp;
label = GS_CCSDSdemodulation_cons(length(RXdata),RXdata,APSKorder,constellation);
label = reshape(label,[],1);

[symnum,ser]=symerr(label,DECdata);
[bitnum,ber]=biterr(label,DECdata);

ber
 
%% plot 
trainloss = importdata('train_loss.txt');
valloss = importdata('val_loss.txt');
epoch = importdata('epoch.txt');
figure
plot(epoch,trainloss,'b','linewidth',1.5);
hold on;
plot(epoch,valloss,'r','linewidth',1.5);
legend('Train loss','Test loss');
xlabel('Epoch');ylabel('MSE');

figure;plot(RXdata,'.');
gscatter(real(RXdata),imag(RXdata),DECdata);hold on
error=RXdata(label~=DECdata);
plot(error,'k.','markersize',8)
