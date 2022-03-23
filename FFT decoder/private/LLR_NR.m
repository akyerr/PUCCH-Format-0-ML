% Approximate LLR. Please see Simplified Soft-Output Demapper for Binary
% Interleaved COFDM with Application to HIPERLAN/2.
function[out_llr] = LLR_NR(Y, mod, nLayers)

% Layer Demap
if nLayers > 4
    cw1 = [];
    for i = 1:ceil(nLayers/2)
        cw1 = [cw1 Y(:,i)];
    end
    cw1 = reshape(cw1.',[],1);
    cw2 = [];
    for i = ceil(nLayers/2)+1:nLayers
        cw2 = [cw2 Y(:,i)];
    end
    cw2 = reshape(cw2.',[],1);
    Y = [cw1 ; cw2];
else
    cw1 = [];
    for i = 1:nLayers
        cw1 = [cw1 Y(:,i)];
    end
    cw1 = reshape(cw1.',[],1);
    Y = cw1;
end

N = length(Y);
yr = real(Y);
yi = imag(Y);
LLR = [];

%% LLR Gen
% After equalization, the input will be scaled to the constellation. We rescale it so that the constellaion points are integers
if (mod == "BPSK")
    LLR = zeros(N,1);
    % bit b(0)
    LLR(:,1) = -yr;
    
elseif (mod == "QPSK")
    % QPSK the scaling factor is sqrt(2);
    yr = yr*sqrt(2);
    yi = yi*sqrt(2);
    LLR = zeros(N,2);
    
    % bit b(0)
    LLR(:,1) = -yr;
    
    % bit b(1)
    LLR(:,2) = -yi;
    
elseif (mod == "16QAM")
    % 16QAM the scaling factor is sqrt(10);
    LLR = zeros(N,4);
    yr = yr*sqrt(10);
    yi = yi*sqrt(10);
    
    for k = 1:N
        % Approx 2
        LLR(k,1) = -yr(k);
        LLR(k,2) = -yi(k);
        LLR(k,3) = abs(yr(k))-2;
        LLR(k,4) = abs(yi(k))-2;
        
        % % Approx 1
        
        % % Bit b(0)
        % if (abs(yr(k)) <= 2)
        % LLR(k,1) = -yr(k);
        % elseif (yr(k) > 2)
        % LLR(k,1) = -2*(yr(k)-1);
        % elseif (yr(k)<-2)
        % LLR(k,1) = -2*(yr(k) + 1);
        % end
        
        % % Bit b(1)
        % if (abs(yi(k)) <= 2)
        % LLR(k,2) = -yi(k);
        % elseif (yi(k) > 2)
        % LLR(k,2) = -2*(yi(k)-1);
        % elseif (yi(k)<-2)
        % LLR(k,2) = -2*(yi(k) + 1);
        % end
        
        % % Bit b(2)
        % LLR(k,3) = abs(yr(k))-2;
        
        % % Bit b(3)
        % LLR(k,4) = abs(yi(k))-2;
    end
    
elseif (mod == "64QAM")
    % 64QAM the scaling factor is sqrt(42);
    LLR = zeros(N,6);
    yr = yr*sqrt(42);
    yi = yi*sqrt(42);
    
    for k = 1:N
        % Approx 2:
        LLR(k,1) = -yr(k);
        LLR(k,2) = -yi(k);
        LLR(k,3) = -(-abs(yr(k)) + 4);
        LLR(k,4) = -(-abs(yi(k)) + 4);
        LLR(k,5) = -(-abs(abs(yr(k))-4) + 2);
        LLR(k,6) = -(-abs(abs(yi(k))-4) + 2);
        
        
        % % Approx 1
        
        % % Bit b(0)
        % if (yr(k) <= -6)
        % LLR(k,1) = -4*(yr(k) + 3);
        % elseif (yr(k)>-6)&(yr(k) <= -4)
        % LLR(k,1) = -3*(yr(k) + 2);
        % elseif (yr(k)>-4)&(yr(k) <= -2)
        % LLR(k,1) = -2*(yr(k) + 1);
        % elseif (abs(yr(k))<2)
        % LLR(k,1) = -yr(k);
        % elseif (2<yr(k))&(yr(k) <= 4)
        % LLR(k,1) = -2*(yr(k)-1);
        % elseif (4<yr(k))&(yr(k) <= 6)
        % LLR(k,1) = -3*(yr(k)-2);
        % elseif (6<yr(k))
        % LLR(k,1) = -4*(yr(k)-3);
        % end
        
        % % Bit b(1)
        % if (yi(k) <= -6)
        % LLR(k,2) = -4*(yi(k) + 3);
        % elseif (yi(k)>-6)&(yi(k) <= -4)
        % LLR(k,2) = -3*(yi(k) + 2);
        % elseif (yi(k)>-4)&(yi(k) <= -2)
        % LLR(k,2) = -2*(yi(k) + 1);
        % elseif (abs(yi(k))<2)
        % LLR(k,2) = -yi(k);
        % elseif (2<yi(k))&(yi(k) <= 4)
        % LLR(k,2) = -2*(yi(k)-1);
        % elseif (4<yi(k))&(yi(k) <= 6)
        % LLR(k,2) = -3*(yi(k)-2);
        % elseif (6<yi(k))
        % LLR(k,2) = -4*(yi(k)-3);
        % end
        
        % % Bit b(2)
        % if (abs(yr(k)) >= 6)
        % LLR(k,3) = -2*(-abs(yr(k)) + 5);
        % elseif (2<abs(yr(k)))&(abs(yr(k)) <= 6)
        % LLR(k,3) = -(4-abs(yr(k)));
        % elseif (abs(yr(k)) <= 2)
        % LLR(k,3) = -2*(-abs(yr(k)) + 3);
        % end
        
        % % Bit b(3)
        % if (abs(yi(k)) >= 6)
        % LLR(k,4) = -2*(-abs(yi(k)) + 5);
        % elseif (2<abs(yi(k)))&(abs(yi(k)) <= 6)
        % LLR(k,4) = -(4-abs(yi(k)));
        % elseif (abs(yi(k)) <= 2)
        % LLR(k,4) = -2*(-abs(yi(k)) + 3);
        % end
        
        % % Bit b(4)
        % if (abs(yr(k)) >4)
        % LLR(k,5) = -(-abs(yr(k)) + 6);
        % elseif (abs(yr(k)) <= 4)
        % LLR(k,5) = -(abs(yr(k))-2);
        % end
        
        % % Bit b(5)
        % if (abs(yi(k)) >4)
        % LLR(k,6) = -(-abs(yi(k)) + 6);
        % elseif (abs(yi(k)) <= 4)
        % LLR(k,6) = -(abs(yi(k))-2);
        % end
    end
    
elseif (mod == "256QAM")
    % 256QAM the scaling factor is sqrt(170);
    LLR = zeros(N,8);
    yr = yr*sqrt(170);
    yi = yi*sqrt(170);
    
    for k = 1:N
        % Approx 2:
        LLR(k,1) = -yr(k);
        LLR(k,2) = -yi(k);
        LLR(k,3) = -(-abs(yr(k)) + 8);
        LLR(k,4) = -(-abs(yi(k)) + 8);
        LLR(k,5) = -(-abs(abs(yr(k))-8) + 4);
        LLR(k,6) = -(-abs(abs(yi(k))-8) + 4);
        LLR(k,7) = -(-abs(-abs(abs(yr(k))-8) + 4 ) + 2);
        LLR(k,8) = -(-abs(-abs(abs(yi(k))-8) + 4 ) + 2);
    end
end

temp = LLR.';
temp = temp(:);
if nLayers > 4
    end_loc = length(temp);
    start_loc = end_loc*ceil(nLayers/2)/nLayers;
    out_llr{1} = temp(1:start_loc);
    out_llr{2} = temp(start_loc+1:end_loc);
else
    out_llr{1} = temp;
end
end
