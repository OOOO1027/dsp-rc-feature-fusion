function [dec_v1rx_PAM6,dec_v1rx_PAM4]=NLTCP_demodulation(datalength,rxdata)

v1rx=rxdata/2+2.5;
rxsymbol=zeros(1,datalength);
     cons=0:5;
     for n=1:datalength
    pointdistance=abs(v1rx(n)-cons);
    rxsymbolnum=pointdistance==min(pointdistance);
    rxsymbol(n)=cons(rxsymbolnum);
     end

dec_v1rx_PAM6 = rxsymbol;

dec_v1rx_PAM4=zeros(1,datalength);
for n=2:datalength
   dec_v1rx_PAM4(n)=abs( rxsymbol(n)-floor(rxsymbol(n-1)/2));
end

end


