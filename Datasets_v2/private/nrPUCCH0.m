function [sym, seqCS] = nrPUCCH0(ack,sr,symAllocation,cp,nslot,nid,groupHopping,initialCS,freqHopping,varargin)
%nrPUCCH0 Physical uplink control channel format 0
%   SYM = nrPUCCH0(ACK,SR,SYMALLOCATION,CP,NSLOT,NID,GROUPHOPPING,INITIALCS,FREQHOPPING)
%   returns the PUCCH format 0 symbols SYM as per TS 38.211 Section
%   6.3.2.3, by considering the following inputs:
%   ACK           - Acknowledgment bits of hybrid automatic repeat request
%                   (HARQ-ACK). It is a column vector of length 0, 1 or 2
%                   HARQ-ACK bits. The bit value of 1 stands for positive
%                   acknowledgment and bit value of 0 stands for negative
%                   acknowledgment. Use empty ([]) to indicate no HARQ-ACK
%                   transmission.
%   SR            - Scheduling request (SR). It is a column vector of
%                   length 0 or 1 SR bits. The bit value of 1 stands for
%                   positive SR and bit value of 0 stands for negative SR.
%                   Use empty ([]) to indicate no SR transmission. The
%                   output SYM is empty when there is only negative SR
%                   transmission.
%   SYMALLOCATION - Symbol allocation for PUCCH transmission. It is a
%                   two-element vector, where first element is the symbol
%                   index corresponding to first OFDM symbol of the PUCCH
%                   transmission in the slot and second element is the
%                   number of OFDM symbols allocated for PUCCH
%                   transmission, which is either 1 or 2.
%   CP            - Cyclic prefix ('normal','extended').
%   NSLOT         - Slot number in radio frame. It is in range 0 to 159 for
%                   normal cyclic prefix for different numerologies. For
%                   extended cyclic prefix, it is in range 0 to 39, as
%                   specified in TS 38.211 Section 4.3.2.
%   NID           - Scrambling identity. It is in range 0 to 1023 if
%                   higher-layer parameter hoppingId is provided, else, it
%                   is in range 0 to 1007, equal to the physical layer cell
%                   identity NCellID.
%   GROUPHOPPING  - Group hopping configuration. It is one of the set
%                   {'neither','enable','disable'} provided by higher-layer
%                   parameter pucch-GroupHopping.
%   INITIALCS     - Initial cyclic shift (m_0). It is in range 0 to 11,
%                   provided by higher-layer parameter initialCyclicShift.
%   FREQHOPPING   - Intra-slot frequency hopping. It is one of the set
%                   {'enabled','disabled'} provided by higher-layer
%                   parameter intraSlotFrequencyHopping.
%
%   The output symbols SYM is of length given by product of number of
%   subcarriers in a resource block and the number of OFDM symbols
%   allocated for PUCCH transmission in SYMALLOCATION.
%
%   Note that when GROUPHOPPING is set to 'disable', sequence hopping is
%   enabled which might result in selecting a sequence number that is not
%   appropriate for short base sequences.
%
%   SYM = nrPUCCH0(...,NAME,VALUE) specifies an additional option as a
%   NAME,VALUE pair to allow control over the format of the symbols:
%
%   'OutputDataType' - 'double' for double precision (default)
%                      'single' for single precision
%
%   Example 1:
%   % Get the PUCCH format 0 symbols for transmitting positive SR when the
%   % starting symbol in a slot is 11, number of PUCCH symbols is 2, slot
%   % number is 63, cell identity is 512, initial cyclic shift is 5, with
%   % normal cyclic prefix, intra-slot frequency hopping disabled and group
%   % hopping enabled.
%
%   ack = [];
%   sr = 1;
%   symAllocation = [11 2];
%   cp = 'normal';
%   nslot = 63;
%   nid = 512;
%   groupHopping = 'enable';
%   initialCS = 5;
%   freqHopping = 'disabled';
%   sym = nrPUCCH0(ack,sr,symAllocation,cp,nslot,nid,groupHopping,initialCS,freqHopping);
%
%   Example 2:
%   % Get the PUCCH format 0 symbols for transmitting 2-bit HARQ-ACK and
%   % negative SR when the starting symbol in a slot is 10, number of PUCCH
%   % symbols is 2, slot number is 3, cell identity is 12, initial cyclic
%   % shift is 5, with extended cyclic prefix, intra-slot frequency hopping
%   % disabled and group hopping enabled.
%
%   ack = [1;1];
%   sr = 0;
%   symAllocation = [10 2];
%   cp = 'extended';
%   nslot = 3;
%   nid = 12;
%   groupHopping = 'enable';
%   initialCS = 5;
%   freqHopping = 'disabled';
%   sym = nrPUCCH0(ack,sr,symAllocation,cp,nslot,nid,groupHopping,initialCS,freqHopping);
%
%   See also nrPUCCH1, nrPUCCH2, nrPUCCH3, nrPUCCH4, nrPUCCHHoppingInfo,
%   nrLowPAPRS.

% Copyright 2018-2020 The MathWorks, Inc.

%#codegen

    coder.extrinsic('nr5g.internal.parseOptions');

    narginchk(9,11);

    % Validate inputs
    fcnName = 'nrPUCCH0';
    [lenACK,lenSR,symStart,nPUCCHSym,cp,groupHopping,freqHopping] = ...
    nr5g.internal.pucch.validatePUCCHInputs(ack,sr,symAllocation,cp,nslot,nid,groupHopping,initialCS,freqHopping,fcnName);

    % Return empty output either for empty inputs or for negative SR
    % transmission only.
    if (lenACK==0) && ((lenSR==0) || (sr(1)==0))
        % Empty sequence
        seq = zeros(0,1);
    else
        % Get the possible cyclic shift values for the length of ack input
        csTable = getCyclicShiftTable(lenACK);

        % Get the sequence cyclic shift based on ack and sr inputs
        if lenACK==0
            seqCS = csTable(1,1);
        elseif (lenSR==0) || (sr(1) ==0)
            uciValue = bi2de(ack','left-msb');
            seqCS = csTable(1,uciValue+1);
        else
            uciValue = bi2de(ack','left-msb');
            seqCS = csTable(2,uciValue+1);
        end

        % Get the hopping parameters
        info = nrPUCCHHoppingInfo(cp,nslot,nid,groupHopping,initialCS,seqCS(1));

        % Get the PUCCH format 0 sequence
        nRBSC = 12;   % Number of subcarriers per resource block
        lps = nrLowPAPRS(info.U(1),info.V(1),info.Alpha(symStart+1:symStart+nPUCCHSym),nRBSC);
        if strcmpi(freqHopping,'enabled') && (nPUCCHSym == 2)
            lps1 = nrLowPAPRS(info.U(2),info.V(2),info.Alpha(symStart+nPUCCHSym),nRBSC);
            seq = [lps(:,1);lps1];
        else
            seq = lps(:);
        end
    end

    % Apply options
    if nargin > 9
        opts = coder.const(nr5g.internal.parseOptions(fcnName,{'OutputDataType'},varargin{:}));
        sym = cast(seq,opts.OutputDataType);
    else
        sym = seq;
    end

end

function csTable = getCyclicShiftTable(len)
%   csTable = getCyclicShiftTable(LEN) provides the possible sequence
%   cyclic shift values based on the length LEN.

    if len == 1
        csTable = [0 6;
                   3 9];
    else
        csTable = [0 3  9 6;
                   1 4 10 7];
    end

end