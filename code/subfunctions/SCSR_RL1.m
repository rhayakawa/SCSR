function [x_est,arrSER] = SCSR_RL1(y,A,arrC,matQ,invMat,lambda,rho,nIteration,x_true);
%SCSR_RL1 SCSR optimization using real L1 norm for complex discrete-valued vector 
%
% Input
%   y: measurement vector
%   A: measurement matrix
%   arrC: array for candidate of unknown variables
%   matQ: coefficients in W-SCSR optimization
%   invMat: inversion matrix in the algorithm
%   lambda, rho: parameters in the algorithm
%   nIteration: number of iterations
%   x_true: true unknown vector (only used in the evalution of SER)
%
% Output
%   x_est: estimate of unknown vector
%   arrSER: array of symbol error rate (SER)
%
    
  [M,N]=size(A);
  L=length(arrC);
  
  x_MF=A'*y;
  Phi=kron(ones(L,1),eye(N));
  thre=reshape(matQ,N*L,1)/(2*rho);
  orig=kron(arrC.',ones(N,1));
  z=zeros(L*N,1);
  r=zeros(L*N,1);
  arrSER=zeros(1,nIteration);
  for k=1:nIteration
    s=invMat*(rho*Phi.'*(z-r)+lambda*x_MF);
    u=Phi*s+r;
    prox_real=sign(real(u-orig)).*max(abs(real(u-orig))-thre,0);
    prox_imag=sign(imag(u-orig)).*max(abs(imag(u-orig))-thre,0);
    z=orig+prox_real+1j*prox_imag;
    r=r+Phi*s-z;
    arrSER(k)=nnz(quantize(s,arrC)-x_true)/N;
  end
  x_est=s;
    
end