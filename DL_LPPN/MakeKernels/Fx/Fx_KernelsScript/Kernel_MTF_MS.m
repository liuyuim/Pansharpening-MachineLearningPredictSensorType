%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           MTF filters the image I_MS using a Gaussin filter matched with the Modulation Transfer Function (MTF) of the MultiSpectral (MS) sensor. 
% 
% Interface:
%           I_Filtered = MTF(I_MS,sensor,tag,ratio)
%
% Inputs:
%           I_MS:           MS image;
%           sensor:         String for type of sensor (e.g. 'WV2', 'IKONOS');
%           tag:            Image tag. Often equal to the field sensor. It makes sense when sensor is 'none'. It indicates the band number
%                           in the latter case;
%           ratio:          Scale ratio between MS and PAN.
%
% Outputs:
%           I_Filtered:     Output filtered MS image.
% 
% References:
%           [Aiazzi06]      B. Aiazzi, L. Alparone, S. Baronti, A. Garzelli, and M. Selva, 揗TF-tailored multiscale fusion of high-resolution MS and Pan imagery,�
%                           Photogrammetric Engineering and Remote Sensing, vol. 72, no. 5, pp. 591�596, May 2006.
%           [Lee10]         J. Lee and C. Lee, 揊ast and efficient panchromatic sharpening,� IEEE Transactions on Geoscience and Remote Sensing, vol. 48, no. 1,
%                           pp. 155�163, January 2010.
%           [Vivone14]      G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, 揂 Critical Comparison Among Pansharpening Algorithms�, 
%                           IEEE Transaction on Geoscience and Remote Sensing, 2014. (Accepted)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% modified by WLG： 20220325 instead of case none with otherwise
% function I_Filtered = MTF(I_MS,sensor,tag,ratio)
function [] = Kernel_MTF_MS(sensor,ratio,N,nBands)

switch sensor
    case 'QB' 
        GNyq = [0.34 0.32 0.30 0.22]; % Band Order: B,G,R,NIR
    case 'IKONOS'
        GNyq = [0.26,0.28,0.29,0.28]; % Band Order: B,G,R,NIR
    case 'GeoEye1'
        GNyq = [0.23,0.23,0.23,0.23]; % Band Order: B,G,R,NIR
%     case 'WV2'
%         GNyq = [0.35 .* ones(1,7), 0.27];
    otherwise
%         if strcmp(tag,'WV2')
%             GNyq = 0.15 .* ones(1,8);
%         else
%             GNyq = 0.29 .* ones(1,size(I_MS,3));
%         end
        GNyq = 0.29 .* ones(1,4);
end


%%% MTF

% N = 41;
% I_MS_LP = zeros(size(I_MS));
% nBands = size(I_MS,3);
% nBands = 4 ;
fcut = 1/ratio;
   
for ii = 1 : nBands
    alpha = sqrt((N*(fcut/2))^2/(-2*log(GNyq(ii))));
    H = fspecial('gaussian', N, alpha);
    Hd = H./max(H(:));
    h = fwind1(Hd,kaiser(N));
%     I_MS_LP(:,:,ii) = imfilter(I_MS(:,:,ii),real(h),'replicate');
end

% I_Filtered= double(I_MS_LP);
ms_kernel = zeros(N,N,4,4);
ms_kernel(:,:,1,1) = h ;
ms_kernel(:,:,2,2) = h ;
ms_kernel(:,:,3,3) = h ;
ms_kernel(:,:,4,4) = h ;
SaveDir = '.';
saveName = fullfile(SaveDir,[sensor,'_ms_kernel.mat']);
save(saveName,'ms_kernel');

end