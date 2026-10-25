function rxdatasym=GS_CCSDSdemodulation_cons(datalength,rxdata,QAMorder,constellation)


rxsymbol=zeros(1,datalength);
rxdatasym=zeros(1,datalength);
symbol=zeros(1,datalength);

namepre = 'CCSDS';
namepost='.txt';
name1 = [namepre,num2str(QAMorder),constellation,namepost];
data1=importdata(name1);
     cons=complex(data1(:,1),data1(:,2));
     for n=1:datalength
    pointdistance=abs(rxdata(n)-cons);
    rxsymbolnum=find(pointdistance==min(pointdistance));
    rxsymbol(n)=cons(rxsymbolnum);
    rxdatasym(n)=rxsymbolnum-1;
    end;     
 
end


