data = load('mpii_human_pose_v1_u12_1.mat');
data = data.RELEASE;

features = containers.Map('KeyType', 'double', 'ValueType', 'any');
file_img_names = fopen('img_names.txt', 'w');

%
for i=1:size(data.annolist, 2)

  i_annolist = data.annolist(i);
  i_act      = data.act(i);
  if( isfield(i_annolist,'annorect') ) 
    for p=1:size(i_annolist.annorect, 2)


      i_person = i_annolist.annorect(p);

      if( isfield(i_person, 'scale') && isfield(i_person, 'objpos') && isfield(i_person, 'annopoints') && isfield(i_act, 'act_id') && isfield(i_annolist, 'image') ) 
        
        if(i_act.act_id>0 && isfield(i_person.annopoints, 'point') ) 

          t_feat = zeros(16,2);

          p_points = i_person.annopoints.point;

          for b=1:size(p_points, 2)
            t_feat(p_points(b).id+1, 1) = (p_points(b).x - i_person.objpos.x) / i_person.scale;
            t_feat(p_points(b).id+1, 2) = (p_points(b).y - i_person.objpos.y) / i_person.scale;
          end
         
          t_feat = t_feat(:);
          t_super = containers.Map('KeyType', 'char', 'ValueType', 'any');
          img_name = i_annolist.image;
          t_super('img_name') = img_name.name;
          t_super('feat') = t_feat;

          if(~isKey(features, i_act.act_id))
            features(i_act.act_id) = containers.Map('KeyType', 'double', 'ValueType', 'any');
          end
          t_samples = features(i_act.act_id);
          t_samples(features(i_act.act_id).Count + 1) = t_super;
        
        end
 
      end

    end

  end
  
end

save('features.mat', 'features')

n_samples = 20;
v_classes = 0;
t_keys = keys(features);

for c=1:features.Count
  t_class = features(t_keys{c});
  if(t_class.Count>n_samples-1) 
    v_classes = v_classes + 1;
  end
end

feat = zeros(v_classes, n_samples, 32);
img_names = containers.Map('KeyType', 'char', 'ValueType', 'any');

v_classes = 0;
for c=1:features.Count
  t_class = features(t_keys{c});
  if(t_class.Count>n_samples-1) 
    v_classes = v_classes + 1;
    for s=1:n_samples
      t_super = t_class(s);
      feat(v_classes, s, :) = t_super('feat');
      img_names([num2str(v_classes), '_', num2str(s)]) = t_super('img_name');
      fprintf(file_img_names, '%d %d %s\n', v_classes, s, t_super('img_name'));
    end 
  end
end

n_feat = reshape(feat, v_classes*n_samples, 32);
t_mean = mean(n_feat);
t_std  = std(n_feat);
t_mean = repmat(t_mean, v_classes*n_samples, 1);
t_std  = repmat(t_std, v_classes*n_samples, 1);

n_feat = (n_feat - t_mean) ./ t_std;

n_feat = reshape(n_feat, v_classes, n_samples, 32);

fclose(file_img_names);
save('n_feat.mat','n_feat')
save('img_names.mat', 'img_names')
