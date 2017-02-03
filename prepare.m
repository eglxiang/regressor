img_h = 112;
img_w = 96;
num_frames_per_video = 40;

cd('Pain_by_OPR');
folders=dir();
[num_folders,~]=size(folders);

for i=3:num_folders
    current_folder=folders(i).name;
    cd(current_folder);
    subfolders=dir();
    [num_subfolders,~]=size(subfolders);
    for j=3:num_subfolders
        current_subfolder=subfolders(j).name;
        cd(current_subfolder);
        files=dir();
        [num_files,~]=size(files);
        % equally spaced
        interval = floor((num_files-2)/num_frames_per_video);
        vidmat = zeros(num_frames_per_video,img_h,img_w);
        for k=1:num_frames_per_video
            index=(k-1)*interval+1;
            current_img = im2double(rgb2gray(imread(files(index+2).name)));
            vidmat(k,:,:)= current_img;
        end
        cd ..
        save(strcat(int2str(j-2),'.mat'), 'vidmat');
    end
    cd ..
end