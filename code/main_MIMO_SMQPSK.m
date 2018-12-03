%%------------------------------------------------------------------------------------
% simulation for IW-SCSR in MIMO signal detection with discrete-valued sparse vector
%%------------------------------------------------------------------------------------

addpath('subfunctions');

N=50; % number of transmit antennas
M=30; % number of receive antennas
nIteration=100; % number of iterations
nSymbolVector=10; % number of unknown vectors per channel realization and SNR
nSample=5; % number of samples of channel matrix
nUpdate=5; % number of weight update
arrSNR=0:2.5:30; % array for signal-to-noise ratio

% probability distribution
L=5;
p=0.8;
arrP=[p (1-p)/4 (1-p)/4 (1-p)/4 (1-p)/4];
arrC=[0 1+1j -1+1j -1-1j 1-1j];
matQ_init=ones(N,1)*arrP;

% parameter of the proposed algorithm
beta=10;
rho=0.1;

% parameter for correlated channels
c=299792458; % [m/s]
frequency=5.0*10^(9); % [Hz]
wavelength=c/frequency; % [m]
r=0.5;
d_t=r*wavelength;
d_r=r*wavelength;

rng('shuffle');

matSumSER_SCSR=zeros(nUpdate,length(arrSNR));
for i=1:nSample
  disp(['i=' num2str(i)]);
  % channel matrix
  % [A,~]=makeChannel(M,N); % i.i.d.
  [A,~]=makeChannel_corr(M,N,d_t,d_r,wavelength);
  
  for SNRIndex=1:length(arrSNR)
    SNR=arrSNR(SNRIndex);
    disp(['  SNR=' num2str(SNR)]);
    % variance of additive noise
    sigma_c=sqrt(2*(1-p)*N/(10^(SNR/10)));
    % parameter of optimization problem
    numer=arrP(1)*sum(sum(matQ_init.*abs(arrC(1)-ones(N,1)*arrC)));
    for l=2:L
      numer=numer+arrP(l)*sum(sum(matQ_init.*(abs(real(arrC(l)-ones(N,1)*arrC))+abs(imag(arrC(l)-ones(N,1)*arrC)))));
    end
    lambda_init=numer/(beta*M*sigma_c^(2));
    % inverse matrix
    invMat_init=(rho*L*eye(N)+lambda_init*(A'*A))^(-1);

    for symbolVectorIndex=1:nSymbolVector
      % transmitted signal vector
      N_active=round((1-p)*N);
      indeces=randperm(N,N_active);
      x=zeros(N,1);
      x(indeces)=(randi([0,1],N_active,1)*2-ones(N_active,1))+1j*(randi([0,1],N_active,1)*2-ones(N_active,1));
      % additive noise vector
      v=(randn(M,1)+1j*randn(M,1))/sqrt(2)*sigma_c;
      % received signal vector
      y=A*x+v;
      
      %% IW-SCSR
      matQ=matQ_init;
      invMat=invMat_init;
      lambda=lambda_init;
      for itrIndex=1:nUpdate
        [x_est,~]=SCSR_SMQPSK(y,A,arrC,matQ,invMat,lambda,rho,nIteration,x);
        matSumSER_SCSR(itrIndex,SNRIndex)=matSumSER_SCSR(itrIndex,SNRIndex)+nnz(quantize(x_est,arrC)-x)/N;
        matD=abs(x_est*ones(1,L)-ones(N,1)*arrC);
        matQ=matD.^(-1)./(sum((matD.^(-1)),2)*ones(1,L));
        % parameter of optimization problem
        numer=arrP(1)*sum(sum(matQ.*abs(arrC(1)-ones(N,1)*arrC)));
        for l=2:L
          numer=numer+arrP(l)*sum(sum(matQ.*(abs(real(arrC(l)-ones(N,1)*arrC))+abs(imag(arrC(l)-ones(N,1)*arrC)))));
        end
        lambda=numer/(beta*M*sigma_c^(2));
        invMat=(rho*L*eye(N)+lambda*(A'*A))^(-1);
      end
    end
  end
end
matSER_SCSR=matSumSER_SCSR/nSample/nSymbolVector

%% Display results
arrMarker=['o';'^';'s';'d';'v';'*';'<';'x';'d';'p';'h';];
close all;
figure;
setLegend={};
for itrIndex=1:4:nUpdate
  h=semilogy(arrSNR,matSER_SCSR(itrIndex,:),['-' arrMarker(itrIndex)],'LineWidth',1,'MarkerSize',8,'MarkerFaceColor','auto');
  setLegend=[setLegend ['IW-SCSR ($T=' num2str(itrIndex) '$)']];
  set(h, 'MarkerFaceColor', get(h,'Color'));
  hold on;
end
grid on;
objLegend=legend(setLegend,'Location','northeast');
objLegend.Interpreter='latex';
objLegend.FontSize=16;
fig=gca;
fig.FontSize=18;
fig.TickLabelInterpreter='latex';
fig.XLabel.Interpreter='latex';
fig.YLabel.Interpreter='latex';
xlabel('SNR (dB)');
ylabel('SER');
axis([arrSNR(1) arrSNR(length(arrSNR)) 1e-5 1]);
saveas(h, 'MIMO_SMQPSK.eps', 'epsc');
