function [Hcomp,H]=makeChannel_corr(m,n,d_t,d_r,lambda)

    Hcomp=(randn(m,n)+1i*randn(m,n))/sqrt(2);
    
    Delta_t=zeros(n,n);
    for i=1:n
        Delta_t(i,:)=abs((i-1):(-1):(i-n));
    end
    Delta_r=zeros(m,m);
    for i=1:m
        Delta_r(i,:)=abs((i-1):(-1):(i-m));
    end
    Phi_t=besselj(0,Delta_t*2*pi*d_t/lambda);
    Phi_r=besselj(0,Delta_r*2*pi*d_r/lambda);
    Hcomp=Phi_r^(1/2)*Hcomp*Phi_t^(1/2);
    
    Hreal=real(Hcomp);
    Himag=imag(Hcomp);
    H=[Hreal -Himag;Himag Hreal];
end