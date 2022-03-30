function audioAug = noise_injection(audioIn, fs, output_path)
    noise_probability = 1;
    SNR_value = 20;
    
    augmenter = audioDataAugmenter( ...
        "AugmentationParameterSource","specify", ...
        "AddNoiseProbability", noise_probability, ...
        "SNR", SNR_value, ...
        "ApplyTimeStretch", false,...
        "ApplyVolumeControl", false, ...
        "ApplyPitchShift", false, ...
        "ApplyTimeStretch", false, ...
        "ApplyTimeShift", false);
    
    
    data = augment(augmenter, audioIn, fs);
    audioAug = data.Audio{1};
    
%     [pathstr, name, ext] = fileparts(audio_path);
%     if isempty(pathstr)
%         pathstr = '.';
%     end
%     output_path = sprintf('%s/%s_noise%s', pathstr, name, ext);
    
    audiowrite(output_path, audioAug, fs);

end