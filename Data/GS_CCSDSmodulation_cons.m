function qamdata=GS_CCSDSmodulation_cons(datalength,decdata,QAMorder,constellation)

symbol=zeros(1,datalength);
namepre = 'CCSDS';
namepost='.txt';
name1 = [namepre,num2str(QAMorder),constellation,namepost];
data1=importdata(name1);
     cons=complex(data1(:,1),data1(:,2));
     for n=1:datalength
    index1=decdata(n)+1;
     qamdata(n)=cons(index1);     
     end

end


