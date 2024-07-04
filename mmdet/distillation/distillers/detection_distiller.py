import torch.nn as nn
import torch.nn.functional as F
import torch
from mmdet3d.models import build_model
from mmcv.runner import  load_checkpoint
from ..builder import DISTILLER,build_distill_loss
from mmdet3d.models.fusion_models import Base3DFusionModel
from mmcv.runner import auto_fp16, force_fp32



@DISTILLER.register_module()
class DetectionDistiller(Base3DFusionModel):
    """Base distiller for detectors.

    It typically consists of teacher_model and student_model.
    """
    def __init__(self,
                 teacher_cfg,
                 student_cfg,
                 distill_cfg=None,**kwargs):

        super().__init__()
        
        self.teacher = build_model(teacher_cfg.model,
                                        train_cfg=teacher_cfg.get('train_cfg'),
                                        test_cfg=teacher_cfg.get('test_cfg'))
        self.init_weights_teacher(kwargs['teacher_pretrained'])

        
        self.teacher.eval()
        
        self.student = build_model(student_cfg.model,
                                        train_cfg=student_cfg.get('train_cfg'),
                                        test_cfg=student_cfg.get('test_cfg'))

        self.init_weights_students(kwargs['student_pretrained'])

        self.distill_losses = nn.ModuleDict()

        self.distill_cfg = distill_cfg
        student_modules = dict(self.student.named_modules())
        teacher_modules = dict(self.teacher.named_modules())
        def regitster_hooks(student_module,teacher_module):
            def hook_teacher_forward(module, input, output):
                
                    self.register_buffer(teacher_module,output)
                
            def hook_student_forward(module, input, output):

                    self.register_buffer( student_module,output )
            return hook_teacher_forward,hook_student_forward
        
        for item_loc in distill_cfg:
            
            student_module = 'student_' + item_loc.student_module.replace('.','_')
            teacher_module = 'teacher_' + item_loc.teacher_module.replace('.','_')

            self.register_buffer(student_module,None)
            self.register_buffer(teacher_module,None)

            hook_teacher_forward,hook_student_forward = regitster_hooks(student_module ,teacher_module )
            teacher_modules[item_loc.teacher_module].register_forward_hook(hook_teacher_forward)
            student_modules[item_loc.student_module].register_forward_hook(hook_student_forward)

            for item_loss in item_loc.methods:
                loss_name = item_loss.name
                self.distill_losses[loss_name] = build_distill_loss(item_loss)
    
    def base_parameters(self):
        return nn.ModuleList([self.student.encoders.camera, self.distill_losses])
    
    def discriminator_parameters(self):
        return self.discriminator

    def init_weights_students(self, path=None):
        checkpoint = load_checkpoint(self.student, path, map_location='cpu')

    def init_weights_teacher(self, path=None):
        """Load the pretrained model in teacher detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        checkpoint = load_checkpoint(self.teacher, path, map_location='cpu')



    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        with torch.no_grad():
            self.teacher.eval()
            features = []
            for sensor in (self.teacher.encoders if self.training else list(self.teacher.encoders.keys())[::-1]) :
                if sensor == 'camera' :
                    feature = self.teacher.extract_camera_features(                    
                            img,
                            points,
                            camera2ego,
                            lidar2ego,
                            lidar2camera,
                            lidar2image,
                            camera_intrinsics,
                            camera2lidar,
                            img_aug_matrix,
                            lidar_aug_matrix,
                            metas)
           
        features = []
        for sensor in (self.student.encoders if self.training else list(self.student.encoders.keys())[::-1]) :
            if sensor == 'camera' :
                feature = self.student.extract_camera_features(                    
                        img,
                        points,
                        camera2ego,
                        lidar2ego,
                        lidar2camera,
                        lidar2image,
                        camera_intrinsics,
                        camera2lidar,
                        img_aug_matrix,
                        lidar_aug_matrix,
                        metas)
            elif sensor == 'lidar' :
                feature = self.student.extract_lidar_features(points)
            features.append(feature)
        if self.student.fuser is not None :
            x = self.student.fuser(features)
        else :
            x = features[0]
        
        x = self.student.decoder["backbone"](x)
        x = self.student.decoder["neck"](x)
        outputs = {}
        for type, head in self.student.heads.items() :
            if type == 'object' :
                pred_dict = head(x, metas)
                losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
            for name, val in losses.items():
                if val.requires_grad:
                    outputs[f"loss/{type}/{name}"] = val * self.student.loss_scale[type]
                else:
                    outputs[f"stats/{type}/{name}"] = val
        
        buffer_dict = dict(self.named_buffers())

        for item_loc in self.distill_cfg:
            
            student_module = 'student_' + item_loc.student_module.replace('.','_')
            teacher_module = 'teacher_' + item_loc.teacher_module.replace('.','_')
            
            student_feat = buffer_dict[student_module]
            teacher_feat = buffer_dict[teacher_module]

            for item_loss in item_loc.methods:
                loss_name = item_loss.name
                
                outputs[loss_name] = self.distill_losses[loss_name](student_feat,teacher_feat)
        
        
        return outputs

