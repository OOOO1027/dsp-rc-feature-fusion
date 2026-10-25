function data_recover1r1 = CAPdemodulation(dataRx,upsampleno,datano)
       
%     roll_off=0.105;
    roll_off=0.105;
    filterlength=10 ; %截断长度
    filtersps=4; %每个符号周期有SPS个采样点
    
    startFreq=1/100;
    % BW=1.2; % 固定spacing
    BW=1+roll_off; %可变spacing
    subcar1=BW/2+startFreq;

    t1=0:2*pi/upsampleno:(datano*upsampleno-1)*2*pi/upsampleno;
    cos1=cos(subcar1*t1);
    sin1=sin(subcar1*t1);

    filter_cos=rcosdesign(roll_off,filterlength,filtersps); %根升余弦滚降滤波器，‘sqrt’均方根升余弦；‘normal’升余弦
    
    offset1=0;
    data_rxI=dataRx.*cos1;
    data_rxQ=dataRx.*sin1;
    data_rx=data_rxI-1i*data_rxQ;
    data_recover1=conv(data_rx,filter_cos, 'same');
    data_recover=downsample(data_recover1,upsampleno,offset1);
    data_recover1r1=data_recover/sqrt(mean(abs(data_recover).^2));
   
end
