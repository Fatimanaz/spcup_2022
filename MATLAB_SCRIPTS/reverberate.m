function audioRev = reverberate(audioIn, fs, output_path)

predelay = 0;
high_cf = 20000;
diffusion = 0.5;
decay = 0.5;
hifreq_damp = 0.9;
wetdry_mix = 0.25;
fsamp = 16000;

reverb = reverberator( ...
	"PreDelay", predelay, ...
	"HighCutFrequency", high_cf, ...
	"Diffusion", diffusion, ...
	"DecayFactor", decay, ...
    "HighFrequencyDamping", hifreq_damp, ...
	"WetDryMix", wetdry_mix, ...
	"SampleRate", fsamp);

audioRev = reverb(audioIn);
% Stereo to mono
audioRev = .5*(audioRev(:,1) + audioRev(:,2));

% [pathstr, name, ext] = fileparts(audio_path);
% if isempty(pathstr)
%     pathstr = '.';
% end
% output_path = sprintf('%s/%s_reverb%s', pathstr, name, ext);

audiowrite(output_path, audioRev, fs);



end