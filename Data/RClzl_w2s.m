%% RC实现从接收波形到直接到符号的解调
close all;
clear;
index = 41;

APSKorder = 64;
constellation='APSK';
upsampleno = 4;
a=0:APSKorder-1;
b=GS_CCSDSmodulation_cons(length(a),a,APSKorder,constellation);
avp=sqrt(mean(abs(b).^2));
dataTx = importdata('data64APSK.txt');
name = ['afterLMS',num2str(index),'.txt'];    % run Rx_LMSVolterra.m first !!!
dataRx = importdata(name);

DECdata = importdata('decimal_64_APSK.txt');
datalen = length(DECdata);
dataTx = reshape(dataTx, 1,[]);
dataRx = reshape(dataRx, 1,[]);
length(DECdata)


dataTx = dataTx/(sqrt(mean(abs(dataTx).^2)));
dataRx = dataRx/(sqrt(mean(abs(dataRx).^2)));

moddata = GS_CCSDSmodulation_cons(length(DECdata),DECdata,APSKorder,constellation);
moddatar = real(moddata);
moddatai = imag(moddata);
label = zeros(length(DECdata), 2);
label(:, 1) = moddatar;
label(:, 2) = moddatai;
label = label.';

%% Metaparameters  （节点数、衰减因子、谱半径）
taps = 25;                    
inlen = taps*upsampleno;      %RC输入的维度(抽头数)
outlen = 2;                

M = 80;                      %储蓄池节点数
gamma = 0.7;                  %衰减

norm = 1e-5;                  %岭回归正则化参数
ratio = 0.1;                  %训练集占比
initiallen = floor(taps/2)+1; %空转长度

rxdata = reshape(dataRx,upsampleno,datalen);
rxdata_pad = zeros(upsampleno,datalen+taps-1);
rxdata_pad(:,(taps-1)/2+1 : (taps-1)/2+datalen) = rxdata;
rx = zeros(taps*upsampleno,datalen);
for i = (taps-1)/2+1:datalen+(taps-1)/2
    temp = rxdata_pad(:,i-((taps-1)/2) : i+(taps-1)/2) ;
    temp = reshape(temp,1,taps*upsampleno);
    rx(:,i-((taps-1)/2)) = temp;
end
rx = rx./max(max(abs(rx)));

rng(100);
A = (rand(M,M)-0.5)*2;        %随机生成邻接矩阵A并根据谱半径sr缩放
sr = abs(max(eig(A))); 
A = A*0.9/sr;
Win = (rand(M,inlen+1)-0.5)*2;%输入连接权重
%% 空转、获取储蓄池状态变量、训练、回归
r = zeros(M,1);
sn = zeros(inlen+1,1);
Ototal_train = zeros(M+inlen+1,datalen*ratio-initiallen); 
Wout = zeros(outlen,M+inlen+1);
Yd_train = label(:, initiallen+1:ratio*datalen);
for n = 1: ratio*datalen
    sn = [rx(:,n);1];
    r = (1-gamma)*r + gamma*tanh( A * r + Win * sn );  %激活函数
    o = [r;rx(:,n);1];
    if n > initiallen
        Ototal_train(:,n-initiallen) = o;
    end
end
I = eye(M+1+inlen);
Wout = Yd_train * Ototal_train' / (Ototal_train * Ototal_train' + norm*I);
Yout_train = Wout * Ototal_train;


moddata_train = Yout_train(1,:) + 1j*Yout_train(2,:);
moddata_train = moddata_train./sqrt(mean(abs(moddata_train).^2)) * avp;
symbol_train = GS_CCSDSdemodulation_cons(length(moddata_train),moddata_train,APSKorder,constellation);
symbol_train = reshape(symbol_train,[],1);
[~,ser_train]=symerr(DECdata(initiallen+1:ratio*datalen),  symbol_train);
[~,ber_train]=biterr(DECdata(initiallen+1:ratio*datalen),  symbol_train);
fprintf('TRAIN:   SER  BER\t%.4e    \t%.4e\n', ser_train, ber_train);

%% 验证集(所有数据)
r = zeros(M,1);
sn = zeros(inlen+1,1);
Ototal_test = zeros(M+inlen+1,datalen-initiallen); 
Yd_test = label(:, initiallen+1:datalen);
for n = 1 : datalen
    sn = [rx(:,n);1];
    r = (1-gamma)*r + gamma*tanh( A * r + Win * sn );
    o = [r;rx(:,n);1];
    if n > (initiallen)
        Ototal_test(:,n-initiallen) = o;
    end
end

Wout = Yd_test * Ototal_test' / (Ototal_test * Ototal_test' + norm*I);
Yout_test = Wout * Ototal_test;

%%  
moddata_test = Yout_test(1,:) + 1j*Yout_test(2,:);
moddata_test = moddata_test./sqrt(mean(abs(moddata_test).^2)) * avp;
symbol_test = GS_CCSDSdemodulation_cons(length(moddata_test),moddata_test,APSKorder,constellation);
symbol_test = reshape(symbol_test,[],1);

[symnum,ser_test]=symerr(DECdata(initiallen+1:datalen),  symbol_test);
[bitnum,ber_test]=biterr(DECdata(initiallen+1:datalen),  symbol_test);
fprintf('TEST:BER\t%.7e\n', ber_test);
recoverdata = moddata_test;
figure;plot(recoverdata,'.');
gscatter(real(recoverdata),imag(recoverdata),DECdata(initiallen+1:datalen));hold on
error=recoverdata(symbol_test~=DECdata(initiallen+1:datalen));
plot(error,'k.','markersize',8)


%%
figure;hold on;grid on;
set(gca,'FontSize',14,'FontName','Arial','Fontweight','bold','linewidth',1.5)
plot(Wout(1,:),'linewidth',1.5);
ylabel('Wout');
box(gca,'on');
xlim([0,220]);
ylim([-18.5,20.5])
% %% save
% sym = [real(recoverdata); imag(recoverdata)];
% 
% r_out = Ototal_test.';
% teacher_sym = Yd_test.';
% 
% namepre = 'BiRC_sym';
% namepost='.txt';
% name=[namepre,num2str(index),namepost];
% eval(['save ' name ' sym  -ASCII']);
% 
% namepre = 'BiRCw2s_Ototal';
% namepost='.txt';
% name=[namepre, num2str(index),namepost];
% eval(['save ' name ' r_out  -ASCII']);
% 
% namepre = 'BiRC_teacher_sym';
% namepost='.txt';
% name=[namepre,num2str(index),namepost];
% eval(['save ' name ' teacher_sym  -ASCII']);
