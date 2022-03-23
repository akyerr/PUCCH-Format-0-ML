tic
clc
clear
close all
dbstop if error

nslot = 13;
SNRdB = [0, 5, 10, 15, 20];
num_iter = 10000;

bit_len_harq = 1;


channel_model = 'fading'; %'fading' or 'awgn'

carrier = nrCarrierConfig;
carrier.NCellID = 0;
carrier.SubcarrierSpacing = 30;
carrier.CyclicPrefix = "normal";
carrier.NSizeGrid = 273;
carrier.NStartGrid = 0;
carrier.NSlot = nslot;
waveformInfo = nrOFDMInfo(carrier);

nFrames = 1;
symbolsPerSlot = carrier.SymbolsPerSlot;
slotsPerFrame = carrier.SlotsPerFrame;
NSlots = nFrames*slotsPerFrame;

nTxAnts = 1;
nRxAnts = 1;

%% Propagation Channel Model Configuration
perfectChannelEstimator = false;
if strcmpi(channel_model, 'fading')
    % Set up TDL channel
    channel = nrTDLChannel;
    channel.DelayProfile = 'TDL-C';
    channel.DelaySpread = 300e-9;
    channel.MaximumDopplerShift = 100; % in Hz
    channel.MIMOCorrelation = 'low';
    channel.TransmissionDirection = 'Uplink';
    channel.NumTransmitAntennas = nTxAnts;
    channel.NumReceiveAntennas = nRxAnts;
    channel.SampleRate = waveformInfo.SampleRate;
    chInfo = info(channel);
    maxChDelay = ceil(max(chInfo.PathDelays*channel.SampleRate));
    maxChDelay = maxChDelay + chInfo.ChannelFilterDelay;
end


%% Set PUCCH format 0 properties
pucch = nrPUCCH0Config;
pucch.NSizeBWP = 273;
pucch.NStartBWP = 0;
pucch.PRBSet = 0;
pucch.SymbolAllocation = [12 2];
pucch.HoppingID = 7;
pucch.InitialCyclicShift = 0;
pucch.FrequencyHopping = "intraSlot";
pucch.SecondHopStartPRB = 272;
pucch.GroupHopping = "neither";
% disp(pucch)

%% Validate PUCCH configuration
classPUCCH = validatestring(class(pucch), {'nrPUCCH0Config', 'nrPUCCH2Config','nrPUCCH3Config','nrPUCCH4Config'},'','class of PUCCH');
formatPUCCH = classPUCCH(8);

accuracy = zeros(length(SNRdB), 1);
for s = 1: length(SNRdB)
    SNR = SNRdB(s);
    if strcmpi(channel_model, 'fading')
        reset(channel);
    end
    success = zeros(num_iter, 1);
    for n = 1: num_iter
        [pucchIndices, pucchIndicesInfo] = nrPUCCHIndices(carrier,pucch);
        ack = randi([0, 1], bit_len_harq, 1);
        sr = 1;
        uci = {ack, sr};
        [pucchSymbols, applied_CS] = nrPUCCH(carrier, pucch, uci);

        % Create resource grid associated with PUCCH transmission antennas
            pucchGrid = nrResourceGrid(carrier, nTxAnts);
            
            % Perform implementation-specific PUCCH MIMO precoding and mapping
            F = eye(1,nTxAnts);
            [~,pucchAntIndices] = nrExtractResources(pucchIndices,pucchGrid);
            pucchGrid(pucchAntIndices) = pucchSymbols*F;
            
            % Perform OFDM modulation
            txWaveform = OFDMModulate(carrier, pucchGrid);
            
            % Apply Channel
            if strcmpi(channel_model, 'fading')
                
                txWaveformChDelay = [txWaveform; zeros(maxChDelay,size(txWaveform,2))];
                [rxWaveform0,pathGains,sampleTimes] = channel(txWaveformChDelay);
            else
                rxWaveform0 = txWaveform;
            end
            
            beta = 1;
            
            % Add Noise
            rxWaveform = awgn(beta*rxWaveform0, SNR);
            
            % CFO
            % rxWaveform = rxWaveform.*exp(-1i*2*pi*(1e3/122.88e6)*(1:length(rxWaveform)).');
            
            rxGrid = OFDMDemodulate(carrier, rxWaveform);
            
            [K,L,R] = size(rxGrid);
            if (L < symbolsPerSlot)
                rxGrid = cat(2,rxGrid,zeros(K,symbolsPerSlot-L,R));
            end
            
            % Extract from actual PUCCH transmission locations
            pucch_rx = nrExtractResources(pucchIndices, rxGrid);
            
%             disp(['Applied cyclic shift at the TX: ', num2str(applied_CS)]);

            recovered_CS = PUCCH0Decode_v0(carrier, pucch, pucch_rx, nslot, pucchSymbols);

%             disp(['TX cyclic shift: ', num2str(applied_CS), 'RX cyclic shift: ', num2str(recovered_CS)]);
            if applied_CS == recovered_CS
                success(n) = 1;
            end
    end
    accuracy(s) = (sum(success)/num_iter)*100;

    disp(['SNR: ', num2str(SNR), 'dB, Accuracy: ', num2str(accuracy(s)), '%']);
end
toc
% Helper Functions
%% OFDM Mod
function tx_out = OFDMModulate(carrier, tx_grid)
SCS = carrier.SubcarrierSpacing;
Nprb = carrier.NSizeGrid;
No_RE = 12*Nprb;
NSymbPerSlot = carrier.SymbolsPerSlot;
num_ant = size(tx_grid, 3);
FFT_size = 2^ceil(log2(No_RE)); % FFT Size
SR = FFT_size*SCS; % Sample rate
slot_samples = SR*1e-3/(SCS/15e3);

CP_Len = 288*FFT_size/4096; % Cyclic prefix length % THE CP LENGTHS ARE DIFFERENT.
CP_Len1 = 352*FFT_size/4096;

ff_ind = 0:No_RE-1;
New_ff_ind = mod(ff_ind-No_RE/2, FFT_size);
RG_mat_final = zeros(FFT_size, NSymbPerSlot, num_ant);
RG_mat_final(New_ff_ind + 1 ,:,:) = tx_grid;


% Now perform IFFT and add the CP.
tx_out = zeros(slot_samples, num_ant);
start_loc = 0;
for l = 1: NSymbPerSlot
    Temp1 = sqrt(FFT_size)*ifft(RG_mat_final(:,l,:), FFT_size);
    
    if (l == 1) % First symbol in a slot has a different CP
        CP_len = CP_Len1;
    else
        CP_len = CP_Len;
    end
    range_loc = 1:(FFT_size+CP_len);
    
    tx_out(start_loc+range_loc,:) = [Temp1(FFT_size - CP_len + 1: FFT_size, 1, :); Temp1];
    start_loc = start_loc+(FFT_size+CP_len);
end
end

%% OFDM Demod
function rxGrid = OFDMDemodulate(carrier, rxWaveform)

% SCS = carrier.SubcarrierSpacing;
Nprb = carrier.NSizeGrid;
No_RE = 12*Nprb;
NSymbPerSlot = carrier.SymbolsPerSlot;
num_ant = size(rxWaveform, 2);
FFT_size = 2^ceil(log2(No_RE)); % FFT Size
% SR = FFT_size*SCS; % Sample rate
% slot_samples = SR*1e-3/(SCS/15e3);

CP_Len = 288*FFT_size/4096; % Cyclic prefix length % THE CP LENGTHS ARE DIFFERENT.
CP_Len1 = 352*FFT_size/4096;

rxGrid = zeros(No_RE, NSymbPerSlot, num_ant);

start_loc = 0;
ff_ind = 0: No_RE-1;
New_ff_ind = mod(ff_ind-No_RE/2, FFT_size);
for l = 1: NSymbPerSlot
    if(l == 1) % CP has a different length for the first symbol
        CP_len = CP_Len1;
    else
        CP_len = CP_Len;
    end
    start_loc = start_loc + CP_len;
    
    % CP removal and FFT
    cp_stripped = rxWaveform(start_loc + (1: FFT_size), :);
    Temp1 = sqrt(1/FFT_size)*fft(cp_stripped, FFT_size);
    
    % Removing the Gaurd Band and doing the inverse mapping to the RB grid.
    rxGrid(:,l,:) = Temp1(New_ff_ind + 1 , :);
    start_loc = start_loc+ FFT_size;
end
end


%% Visualize Grid
function visualize_grid(in_grid)
figure()
imagesc(abs(in_grid(:, :, 1)));
axis xy;
xlabel('OFDM Symbols');
ylabel('Subcarriers');
xticks(1: 14)
xticklabels({'symb 0','symb 1','symb 2', 'symb 3','symb 4','symb 5', ...
    'symb 6','symb 7','symb 8', 'symb 9','symb 10','symb 11', ...
    'symb 12','symb 13'})
hold on
for i = 1:14
    line([i-0.5 i-0.5],[0 size(in_grid, 1)],'Color','white');
end
end



function recovered_CS = PUCCH0Decode_v0(carrier, pucch, sym, slot, pucchSymbols)

%% Setup parameters
CP_type = carrier.CyclicPrefix;
hopping_ID = pucch.HoppingID;
NSymb = pucch.SymbolAllocation(2);
start_symb = pucch.SymbolAllocation(1);
group_hopping = pucch.GroupHopping;
freq_hopping = pucch.FrequencyHopping;
initial_CS = pucch.InitialCyclicShift;

tx_uci_seq = pucchSymbols(1: 12); %#ok
rx_uci_seq = sym(1: 12, 1);

if strcmpi(CP_type, 'normal')
    NSymbSlot = 14;
else
    NSymbSlot = 12;
end

NID = hopping_ID;
v = zeros(1, 2);
fgh = zeros(1, 2);

if strcmpi(group_hopping, 'neither') % No group hopping, no sequence hopping
    % do nothing
elseif strcmpi(group_hopping, 'enable') % Group hopping

    cinit = floor(NID/30);

    % For the first hop we need c(16*slot+m) and for the second hop we
    % need c(16*slot+8+m). Where m = 0 to 7. 8*2*slot is where we
    % start extracting from. 16 is how many elements we extract.

    c = pseudorandom_sequence(cinit, [8*2*slot, 16]);

    % Reshape gives 2 columns. Oen for each hop.
    fgh(1,:) = mod((2.^(0:7))*reshape(c, 8, []), 30);

else % sequence hopping

    % if group hopping is disabled, sequence hopping is enabled within
    % the group.
    cinit = 32*floor(NID/30) + mod(NID,30);

    % 2*slot is where we start extracting from. 2 is no. of elements to
    % extract

    c = pseudorandom_sequence(cinit, [2*slot, 2]);

    v(1,:) = double(c');
end

% Sequence shift offset Mod 30 because there are 30 groups (0 to 29)
% This is for all cases. group hopping enable, disable and neither
fss = mod(NID,30);

% Group numbers. First element: first hop in the slot. Second
% element: second hop in the slot (0 to 29)
u = mod(fgh+fss,30);

c0 = nrPRBS(NID, [NSymbSlot*8*slot, NSymbSlot*8]);
c = pseudorandom_sequence(NID, [NSymbSlot*8*slot, NSymbSlot*8]);

check = sum(xor(c0, c)); %#ok

m0 = initial_CS;

% ncs for all the symbols in a slot
ncs_0 = (2.^(0:7))*reshape(c, 8, []);

% m11 for all the symbols in a slot
m11_0 = mod(m0 + ncs_0, 12); % for all 14 symbols

% extract ncs only for PUCCH symbols in the slot
ncs = ncs_0(start_symb+1: start_symb+NSymb);

% extract cyclic shifts only for the PUCCH symbols in the slot
m11 = m11_0(start_symb+1: start_symb+NSymb)+1;

% base sequece for all PUCCH symbols in the slot (both hops)
base_seq = low_papr_seq(u, v, 12);

base_seq_symb1 = base_seq(:, 1);


x = rx_uci_seq;
y = base_seq_symb1;

[val, ind] = max(abs(fft(x.*conj(y))));

recovered_CS = mod((ind - m11(1)), 12); % first hop only

end

%% Low PAPR Seq
function base_seq = low_papr_seq(u, v, m)
for vid = 1: length(v)
    for uid = 1: length(u)
        %         nIndex = (0: m-1)';
        phi = getPhiType1(u(uid), m);
        base_seq0 = exp(1j.*phi.*pi/4);
    end
end

base_seq = repmat(base_seq0, 1 , length(u));
end

%% Phi
function phi = getPhiType1(u,m)

%   PHI = getPhiType1(U,M) provides the phase values, PHI, to be applied
%   for generating the type 1 base sequence based on the group number U and
%   the sequence length M, as stated in the TS 38.211 Section 5.2.2.

% Get the table of phase values based on the length M
if m == 6 % Table 5.2.2.2-1
    phiTable = [-3  -1   3   3  -1  -3; ...
        -3   3  -1  -1   3  -3; ...
        -3  -3  -3   3   1  -3; ...
        1   1   1   3  -1  -3; ...
        1   1   1  -3  -1   3; ...
        -3   1  -1  -3  -3  -3; ...
        -3   1   3  -3  -3  -3; ...
        -3  -1   1  -3   1  -1; ...
        -3  -1  -3   1  -3  -3; ...
        -3  -3   1  -3   3  -3; ...
        -3   1   3   1  -3  -3; ...
        -3  -1  -3   1   1  -3; ...
        1   1   3  -1  -3   3; ...
        1   1   3   3  -1   3; ...
        1   1   1  -3   3  -1; ...
        1   1   1  -1   3  -3; ...
        -3  -1  -1  -1   3  -1; ...
        -3  -3  -1   1  -1  -3; ...
        -3  -3  -3   1  -3  -1; ...
        -3   1   1  -3  -1  -3; ...
        -3   3  -3   1   1  -3; ...
        -3   1  -3  -3  -3  -1; ...
        1   1  -3   3   1   3; ...
        1   1  -3  -3   1  -3; ...
        1   1   3  -1   3   3; ...
        1   1  -3   1   3   3; ...
        1   1  -1  -1   3  -1; ...
        1   1  -1   3  -1  -1; ...
        1   1  -1   3  -3  -1; ...
        1   1  -3   1  -1  -1];
elseif m == 12 % Table 5.2.2.2-2
    phiTable = [-3   1  -3  -3  -3   3  -3  -1   1   1   1  -3; ...
        -3   3   1  -3   1   3  -1  -1   1   3   3   3; ...
        -3   3   3   1  -3   3  -1   1   3  -3   3  -3; ...
        -3  -3  -1   3   3   3  -3   3  -3   1  -1  -3; ...
        -3  -1  -1   1   3   1   1  -1   1  -1  -3   1; ...
        -3  -3   3   1  -3  -3  -3  -1   3  -1   1   3; ...
        1  -1   3  -1  -1  -1  -3  -1   1   1   1  -3; ...
        -1  -3   3  -1  -3  -3  -3  -1   1  -1   1  -3; ...
        -3  -1   3   1  -3  -1  -3   3   1   3   3   1; ...
        -3  -1  -1  -3  -3  -1  -3   3   1   3  -1  -3; ...
        -3   3  -3   3   3  -3  -1  -1   3   3   1  -3; ...
        -3  -1  -3  -1  -1  -3   3   3  -1  -1   1  -3; ...
        -3  -1   3  -3  -3  -1  -3   1  -1  -3   3   3; ...
        -3   1  -1  -1   3   3  -3  -1  -1  -3  -1  -3; ...
        1   3  -3   1   3   3   3   1  -1   1  -1   3; ...
        -3   1   3  -1  -1  -3  -3  -1  -1   3   1  -3; ...
        -1  -1  -1  -1   1  -3  -1   3   3  -1  -3   1; ...
        -1   1   1  -1   1   3   3  -1  -1  -3   1  -3; ...
        -3   1   3   3  -1  -1  -3   3   3  -3   3  -3; ...
        -3  -3   3  -3  -1   3   3   3  -1  -3   1  -3; ...
        3   1   3   1   3  -3  -1   1   3   1  -1  -3; ...
        -3   3   1   3  -3   1   1   1   1   3  -3   3; ...
        -3   3   3   3  -1  -3  -3  -1  -3   1   3  -3; ...
        3  -1  -3   3  -3  -1   3   3   3  -3  -1  -3; ...
        -3  -1   1  -3   1   3   3   3  -1  -3   3   3; ...
        -3   3   1  -1   3   3  -3   1  -1   1  -1   1; ...
        -1   1   3  -3   1  -1   1  -1  -1  -3   1  -1; ...
        -3  -3   3   3   3  -3  -1   1  -3   3   1  -3; ...
        1  -1   3   1   1  -1  -1  -1   1   3  -3   1; ...
        -3   3  -3   3  -3  -3   3  -1  -1   1   3  -3];
elseif m == 18 % Table 5.2.2.2-3
    phiTable = [-1   3  -1  -3   3   1  -3  -1   3  -3  -1  -1   1   1   1  -1  -1  -1; ...
        3  -3   3  -1   1   3  -3  -1  -3  -3  -1  -3   3   1  -1   3  -3   3; ...
        -3   3   1  -1  -1   3  -3  -1   1   1   1   1   1  -1   3  -1  -3  -1; ...
        -3  -3   3   3   3   1  -3   1   3   3   1  -3  -3   3  -1  -3  -1   1; ...
        1   1  -1  -1  -3  -1   1  -3  -3  -3   1  -3  -1  -1   1  -1   3   1; ...
        3  -3   1   1   3  -1   1  -1  -1  -3   1   1  -1   3   3  -3   3  -1; ...
        -3   3  -1   1   3   1  -3  -1   1   1  -3   1   3   3  -1  -3  -3  -3; ...
        1   1  -3   3   3   1   3  -3   3  -1   1   1  -1   1  -3  -3  -1   3; ...
        -3   1  -3  -3   1  -3  -3   3   1  -3  -1  -3  -3  -3  -1   1   1   3; ...
        3  -1   3   1  -3  -3  -1   1  -3  -3   3   3   3   1   3  -3   3  -3; ...
        -3  -3  -3   1  -3   3   1   1   3  -3  -3   1   3  -1   3  -3  -3   3; ...
        -3  -3   3   3   3  -1  -1  -3  -1  -1  -1   3   1  -3  -3  -1   3  -1; ...
        -3  -1  -3  -3   1   1  -1  -3  -1  -3  -1  -1   3   3  -1   3   1   3; ...
        1   1  -3  -3  -3  -3   1   3  -3   3   3   1  -3  -1   3  -1  -3   1; ...
        -3   3  -1  -3  -1  -3   1   1  -3  -3  -1  -1   3  -3   1   3   1   1; ...
        3   1  -3   1  -3   3   3  -1  -3  -3  -1  -3  -3   3  -3  -1   1   3; ...
        -3  -1  -3  -1  -3   1   3  -3  -1   3   3   3   1  -1  -3   3  -1  -3; ...
        -3  -1   3   3  -1   3  -1  -3  -1   1  -1  -3  -1  -1  -1   3   3   1; ...
        -3   1  -3  -1  -1   3   1  -3  -3  -3  -1  -3  -3   1   1   1  -1  -1; ...
        3   3   3  -3  -1  -3  -1   3  -1   1  -1  -3   1  -3  -3  -1   3   3; ...
        -3   1   1  -3   1   1   3  -3  -1  -3  -1   3  -3   3  -1  -1  -1  -3; ...
        1  -3  -1  -3   3   3  -1  -3   1  -3  -3  -1  -3  -1   1   3   3   3; ...
        -3  -3   1  -1  -1   1   1  -3  -1   3   3   3   3  -1   3   1   3   1; ...
        3  -1  -3   1  -3  -3  -3   3   3  -1   1  -3  -1   3   1   1   3   3; ...
        3  -1  -1   1  -3  -1  -3  -1  -3  -3  -1  -3   1   1   1  -3  -3   3; ...
        -3  -3   1  -3   3   3   3  -1   3   1   1  -3  -3  -3   3  -3  -1  -1; ...
        -3  -1  -1  -3   1  -3   3  -1  -1  -3   3   3  -3  -1   3  -1  -1  -1; ...
        -3  -3   3   3  -3   1   3  -1  -3   1  -1  -3   3  -3  -1  -1  -1   3; ...
        -1  -3   1  -3  -3  -3   1   1   3   3  -3   3   3  -3  -1   3  -3   1; ...
        -3   3   1  -1  -1  -1  -1   1  -1   3   3  -3  -1   1   3  -1   3  -1];
else % m is equal to 24. Table 5.2.2.2-4
    phiTable = [-1  -3   3  -1   3   1   3  -1   1  -3  -1  -3  -1   1   3  -3  -1  -3   3   3   3  -3  -3  -3; ...
        -1  -3   3   1   1  -3   1  -3  -3   1  -3  -1  -1   3  -3   3   3   3  -3   1   3   3  -3  -3; ...
        -1  -3  -3   1  -1  -1  -3   1   3  -1  -3  -1  -1  -3   1   1   3   1  -3  -1  -1   3  -3  -3; ...
        1  -3   3  -1  -3  -1   3   3   1  -1   1   1   3  -3  -1  -3  -3  -3  -1   3  -3  -1  -3  -3; ...
        -1   3  -3  -3  -1   3  -1  -1   1   3   1   3  -1  -1  -3   1   3   1  -1  -3   1  -1  -3  -3; ...
        -3  -1   1  -3  -3   1   1  -3   3  -1  -1  -3   1   3   1  -1  -3  -1  -3   1  -3  -3  -3  -3; ...
        -3   3   1   3  -1   1  -3   1  -3   1  -1  -3  -1  -3  -3  -3  -3  -1  -1  -1   1   1  -3  -3; ...
        -3   1   3  -1   1  -1   3  -3   3  -1  -3  -1  -3   3  -1  -1  -1  -3  -1  -1  -3   3   3  -3; ...
        -3   1  -3   3  -1  -1  -1  -3   3   1  -1  -3  -1   1   3  -1   1  -1   1  -3  -3  -3  -3  -3; ...
        1   1  -1  -3  -1   1   1  -3   1  -1   1  -3   3  -3  -3   3  -1  -3   1   3  -3   1  -3  -3; ...
        -3  -3  -3  -1   3  -3   3   1   3   1  -3  -1  -1  -3   1   1   3   1  -1  -3   3   1   3  -3; ...
        -3   3  -1   3   1  -1  -1  -1   3   3   1   1   1   3   3   1  -3  -3  -1   1  -3   1   3  -3; ...
        3  -3   3  -1  -3   1   3   1  -1  -1  -3  -1   3  -3   3  -1  -1   3   3  -3  -3   3  -3  -3; ...
        -3   3  -1   3  -1   3   3   1   1  -3   1   3  -3   3  -3  -3  -1   1   3  -3  -1  -1  -3  -3; ...
        -3   1  -3  -1  -1   3   1   3  -3   1  -1   3   3  -1  -3   3  -3  -1  -1  -3  -3  -3   3  -3; ...
        -3  -1  -1  -3   1  -3  -3  -1  -1   3  -1   1  -1   3   1  -3  -1   3   1   1  -1  -1  -3  -3; ...
        -3  -3   1  -1   3   3  -3  -1   1  -1  -1   1   1  -1  -1   3  -3   1  -3   1  -1  -1  -1  -3; ...
        3  -1   3  -1   1  -3   1   1  -3  -3   3  -3  -1  -1  -1  -1  -1  -3  -3  -1   1   1  -3  -3; ...
        -3   1  -3   1  -3  -3   1  -3   1  -3  -3  -3  -3  -3   1  -3  -3   1   1  -3   1   1  -3  -3; ...
        -3  -3   3   3   1  -1  -1  -1   1  -3  -1   1  -1   3  -3  -1  -3  -1  -1   1  -3   3  -1  -3; ...
        -3  -3  -1  -1  -1  -3   1  -1  -3  -1   3  -3   1  -3   3  -3   3   3   1  -1  -1   1  -3  -3; ...
        3  -1   1  -1   3  -3   1   1   3  -1  -3   3   1  -3   3  -1  -1  -1  -1   1  -3  -3  -3  -3; ...
        -3   1  -3   3  -3   1  -3   3   1  -1  -3  -1  -3  -3  -3  -3   1   3  -1   1   3   3   3  -3; ...
        -3  -1   1  -3  -1  -1   1   1   1   3   3  -1   1  -1   1  -1  -1  -3  -3  -3   3   1  -1  -3; ...
        -3   3  -1  -3  -1  -1  -1   3  -1  -1   3  -3  -1   3  -3   3  -3  -1   3   1   1  -1  -3  -3; ...
        -3   1  -1  -3  -3  -1   1  -3  -1  -3   1   1  -1   1   1   3   3   3  -1   1  -1   1  -1  -3; ...
        -1   3  -1  -1   3   3  -1  -1  -1   3  -1  -3   1   3   1   1  -3  -3  -3  -1  -3  -1  -3  -3; ...
        3  -3  -3  -1   3   3  -3  -1   3   1   1   1   3  -1   3  -3  -1   3  -1   3   1  -1  -3  -3; ...
        -3   1  -3   1  -3   1   1   3   1  -3  -3  -1   1   3  -1  -3   3   1  -1  -3  -3  -3  -3  -3; ...
        3  -3  -1   1   3  -1  -1  -3  -1   3  -1  -3  -1  -3   3  -1   3   1   1  -3   3  -3  -3  -3];
end

% Get the phase values specific to group number u from the table of
% phase values
phi = phiTable(u+1,:)';

end
