function [Hcomp,H]=makeChannel_CE(m,n,L_path,nBlock)

    Hcomp=zeros(m*nBlock,n*nBlock);
    for i=1:L_path
      H_IR=(randn(m,n)+1i*randn(m,n))/sqrt(2);
      v=[zeros(i-1,1); 1; zeros(nBlock-i,1)];
      matToeplitz=toeplitz(v,[v(1) fliplr(v(2:end)')]);
      Hcomp=Hcomp+kron(matToeplitz,H_IR);
    end

    Hreal=real(Hcomp);
    Himag=imag(Hcomp);
    H=[Hreal -Himag;Himag Hreal];
end