import argparse
import ast
from math import sqrt

from utils import paths

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Build a video relation detection network')

    parser.add_argument('--dataset', dest='dataset',
                        help='The name of dataset to be used',
                        type=str, default='vidvrd')
    parser.add_argument('--clip_len', dest='clip_len',
                        help='Atomatic clip length in training and test',
                        type=int, default=30)
    parser.add_argument('--normalize_visual_feats', dest='normalize_visual_feats',
                        help="L2-normalise region features before matching them "
                             "against the CLIP text embeddings in the object "
                             "classifier. Off by default, matching upstream and the "
                             "un-normalised object bank the checkpoints were trained "
                             "with. Turning it on changes every object score.",
                        action='store_true')
    parser.add_argument('--frame_stride', dest='frame_stride',
                        help="Encode every Nth frame and reuse those features for "
                             "the frames in between. 1 (the default) reproduces the "
                             "shipped EOV behaviour: every frame is encoded. The "
                             "paper's Implementation Details say frames are sampled "
                             "every 30, which is --frame_stride 30 and ~30x cheaper. "
                             "The two do NOT give identical results and it is unknown "
                             "which the released checkpoints were trained with -- see "
                             "docs/10_Known-issues.md before changing it.",
                        type=int, default=1)
    parser.add_argument('--use_unlabeld_pair', dest='use_unlabeld_pair',
                        help='Whether to use unlabeled trajectory pairs for training',
                        action='store_true')                                                    
    parser.add_argument('--batch_size', dest='batch_size',
                        help='Batch size in training',
                        type=int, default=1)
    parser.add_argument('--batch_size_eval', dest='batch_size_eval',
                        help='Batch size in evluation',
                        type=int, default=32)
    parser.add_argument('--lr', dest='lr',
                        help='Initial learning rate',
                        type=float, default=1e-5)
    parser.add_argument('--dropout_r', dest='dropout_r',
                        help='Dropout value',
                        type=float, default=0.5)
    parser.add_argument('--num_layers', dest='num_layers',
                        help='Lstm Number of Layers',
                        type=int, default=0)
    parser.add_argument('--momentum', dest='momentum',
                        help='The momentum of SGD optimizer',
                        type=float, default=0.9)
    parser.add_argument('--weight_decay', dest='weight_decay',
                        help='Model weight decay value',
                        type=float, default=0.0001)
    parser.add_argument('--max_epoch', dest='max_epoch',
                        help='The epoch to stop',
                        type=int, default=20)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='The epoch to run from',
                        type=int, default=0)
    # RB1: was a bare store_true that nothing read. It now takes the path of an
    # epoch checkpoint written by cli/train.py (--epoch_ckpt_dir); a bare
    # `--resume`, or none, means no resume, as before.
    parser.add_argument('--resume', dest='resume',
                        help='Path of an epoch checkpoint to resume from',
                        nargs='?', const='', default='', type=str)
    parser.add_argument('--ckpt_path', dest='ckpt_path',
                        help='The checkpoint saved path',
                        type=str, default="")
    parser.add_argument('--print_freq', dest='print_freq',
                        help='The batch frequence of printing info',
                        type=int, default=100)
    parser.add_argument('--clip_top_n', dest='clip_top_n',
                        help='The top n predictions of clip to be saved',
                        type=int, default=20)
    parser.add_argument('--max_per_video', dest='max_per_video',
                        help='Max number of relations for each video to be saved',
                        type=int, default=200)
    parser.add_argument('--ps', dest='ps',
                        help='The P.S. information for this training process',
                        type=str, default='test_new')
    parser.add_argument('--train_traj', dest='train_traj',
                        help='The trajectories for training split',
                        type=str, default="gt")
    parser.add_argument('--val_traj', dest='val_traj',
                        help='The trajectories source for validation split',
                        type=str, default="gt")
    parser.add_argument('--test_traj', dest='test_traj',
                        help='The trajectories source for testing split',
                        type=str, default="gt")
    parser.add_argument('--use_prior', dest='use_prior',
                        help='Wether to use prior or not',
                        action='store_true')
    parser.add_argument('--temp_model', dest='temp_model',
                        help='The temporal model used to encoding context',
                        default='none')
    parser.add_argument('--obj_loss_weight', dest='obj_loss_weight',
                        help='The loss weight factor for object loss',
                        type=float, default=0.2)
    parser.add_argument('--int_loss_weight', dest='int_loss_weight',
                        help='The loss weight factor for interactive loss',
                        type=float, default=0.1)

    parser.add_argument('--rel_feat', dest='rel_feat',
                        help='Use relative location feature or not',
                        type=ast.literal_eval, default=True)
    parser.add_argument('--mask_feat', dest='mask_feat',
                        help='Use mask location feature or not',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--lan_feat', dest='lan_feat',
                        help='Use language feature or not',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--v2d_feat', dest='v2d_feat',
                        help='Use visual 2d feature or not',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--mot_feat', dest='mot_feat',
                        help='Use motion location feature or not',
                        type=ast.literal_eval, default=True)
    parser.add_argument('--v3d_feat', dest='v3d_feat',
                        help='Use visual 3d feature or not',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--clip_feat', dest='clip_feat',
                        help='Use clip visual feature or not',
                        type=ast.literal_eval, default=True)
    parser.add_argument('--intern_feat', dest='intern_feat',
                        help='Use intern visual feature or not',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--bbox_feat', dest='bbox_feat',
                        help='Use bbox location feature or not',
                        type=ast.literal_eval, default=True)
    parser.add_argument('--ptm_mode', dest='ptm_mode',
                        help='Use vision only or vision-text model to train',
                        type=str, default='vision_text')
    parser.add_argument('--src_split', dest='src_split',
                        help='Use what data for training',
                        type=str, default='base')
    parser.add_argument('--tgt_split', dest='tgt_split',
                        help='Use what data for evaluation',
                        type=str, default='all')

    parser.add_argument('--rel_emb_dim', dest='rel_emb_dim',
                        help='The dimension of relative location feature',
                        type=int, default=256)
    parser.add_argument('--mask_emb_dim', dest='mask_emb_dim',
                        help='The dimension of mask location feature',
                        type=int, default=256)
    parser.add_argument('--lan_emb_dim', dest='lan_emb_dim',
                        help='The dimension of language feature',
                        type=int, default=256)
    parser.add_argument('--v2d_emb_dim', dest='v2d_emb_dim',
                        help='The dimension of visual 2d feature',
                        type=int, default=512)
    parser.add_argument('--mot_emb_dim', dest='mot_emb_dim',
                        help='The dimension of mition location feature',
                        type=int, default=256)
    parser.add_argument('--v3d_emb_dim', dest='v3d_emb_dim',
                        help='The dimension of visual 3d feature',
                        type=int, default=512)
    parser.add_argument('--clip_hidden_dim', dest='clip_hidden_dim',
                        help='The dimension of clip embedding hidden layer',
                        type=int, default=1024)
    parser.add_argument('--clip_emb_dim', dest='clip_emb_dim',
                        help='The dimension of clip embedding output layer',
                        type=int, default=512)
    parser.add_argument('--temp_out_dim', dest='temp_out_dim',
                        help='The dimension of temporal unit hidden layer',
                        type=int, default=512)     
    
    #Prompt-OV
    parser.add_argument('--img_size', default=336, type=int)
    parser.add_argument('--eval_size', default=336, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm') 

    parser.add_argument('--set_cost_class', default=3, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class") 
    parser.add_argument('--dataset_file', default='open_vidvrd')
    parser.add_argument('--vidvrd_path', default=paths.DATA_ROOT, type=str)
    parser.add_argument('--remove_difficult', action='store_true')     
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')    
    parser.add_argument('--eval', default=True, type=lambda x: (str(x).lower() == 'true'), help='eval mode')
    # NOTE: --eval above is a *detector* flag (read in methods/detectors/ov_prompt/solq.py)
    # and defaults to True, so it cannot mean "skip training". This is the
    # evaluate-only switch for cli/train.py.
    parser.add_argument('--eval_only', action='store_true',
                        help='skip training: load --ckpt_path and evaluate once')
    # The step-1/2/3 weights. Defaults are the authors' own filenames; override
    # these rather than renaming your copies.
    parser.add_argument('--detector_ckpt', type=str, default=paths.DETECTOR_CKPT,
                        help='step-1 trajectory detector weights')
    parser.add_argument('--obj_classifier_ckpt', type=str, default=paths.OBJ_CLASSIFIER_CKPT,
                        help='step-2 object classifier weights')
    parser.add_argument('--relation_ckpt', type=str, default=paths.RELATION_CKPT,
                        help='step-3 relationship classifier weights')
    parser.add_argument('--position_embedding', default='sine', type=str)
    parser.add_argument('--pos_dim', default=256, type=int, help="Size of the embeeding for pos")
    parser.add_argument('--reduced_dim', default=256, type=int, help="Size of the embeddings for head")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int, # Deform-DETR: 1024, DETR: 2048
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--dropout_d', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=3, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    parser.add_argument('--method', default='ov_prompt', type=str, help='method names in {vidt, vidt_wo_neck}')
    parser.add_argument("--det_token_num", default=300, type=int, help="Number of det token in the body backbone")

    # * Auxiliary Techniques
    parser.add_argument('--aux_loss', default=True, type=lambda x: (str(x).lower() == 'true'), help='auxiliary decoding loss')
    parser.add_argument('--with_box_refine', default=True, type=lambda x: (str(x).lower() == 'true'), help='iterative box refinement')
    parser.add_argument('--cross_scale_fusion', default=True, type=lambda x: (str(x).lower() == 'true'), help='use of scale fusion')
    parser.add_argument('--n_iter_to_acc', default=1, type=int, help='gradient accumulation step size')

    # * ov detr [default setup]
    parser.add_argument('--cache_mode', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='enable cache mode for data loading')
    parser.add_argument("--label_map", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='enable label map')
    parser.add_argument("--dilation", default=False, type=lambda x: (str(x).lower() == 'true'))

    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument("--enc_n_points", default=4, type=int)
    parser.add_argument("--two_stage", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--max_len", default=15, type=int)
    parser.add_argument("--feature_loss_coef", default=2, type=float) # -> 32 64.
    parser.add_argument("--clip_feat_path", default=paths.CLIP_FEAT_BANK, type=str,)
    parser.add_argument("--prob", default=0.75, type=float)
    parser.add_argument("--amp", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--eval_period", default=1, type=int)
    parser.add_argument("--distil_clip_embed", default=True, type=lambda x: (str(x).lower() == 'true'))

    # segmentor
    parser.add_argument('--masks', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--with_vector', default=False, type=lambda x: (str(x).lower() == 'true'))

    ## New losses
    # iou-aware
    parser.add_argument('--iou_aware', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='use of iou-aware loss')
    parser.add_argument('--iouaware_loss_coef', default=2, type=float)
    parser.add_argument('--new_loss', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='use of new loss')
    parser.add_argument('--new_loss_coef', default=3, type=float)
    # token label
    parser.add_argument('--return_masks', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--token_label', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='use of token label loss')
    parser.add_argument('--token_loss_coef', default=2, type=float)
    parser.add_argument('--all_train_token', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--clip_backbone', default='ViT-L/14@336px', type=str)
    parser.add_argument('--temperature', default=0.01, type=float)
    parser.add_argument('--clip_h', default=336, type=int)
    parser.add_argument('--clip_w', default=336, type=int)
    parser.add_argument('--bg', default=True, type=lambda x: (str(x).lower() == 'true'))

    parser.add_argument('--use_adapter', default=True, type=bool)
    parser.add_argument('--vector_hidden_dim', default=1024, type=int)
    parser.add_argument('--vector_loss_coef', default=3, type=float)
    parser.add_argument('--n_keep', default=256, type=int,
                        help="Number of coeffs to be remained")
    parser.add_argument('--gt_mask_len', default=128, type=int,
                        help="Size of target mask")
    parser.add_argument('--no_vector_loss_norm', default=False, action='store_true')
    parser.add_argument('--activation', default='relu', type=str, help="Activation function to use")
    parser.add_argument('--vector_start_stage', default=0, type=int)
    parser.add_argument('--loss_type', default='l1', type=str)
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    
    parser.add_argument('--train_split', dest='train_split',
                        help='train_split',
                        type=str, default="base")
    parser.add_argument('--test_split', dest='test_split',
                        help='test_split',
                        type=str, default="all")
    parser.add_argument('--N_CTX', dest='N_CTX',
                        help='number of context',
                        type=int, default=16)
    parser.add_argument(
            '--BoT',dest='BoT',help='Replacing the original feature extractor with BoT',
            type=bool, default=True)
    parser.add_argument(
            '--ECC',dest='ECC',
            type=bool,
            default=True,
            help='CMC model'
        )
            
    parser.add_argument(
            '--NSA',dest='NSA',type=bool,
            default=True,
            help='NSA Kalman filter'
        )
    parser.add_argument(
            '--EMA',dest='EMA',type=bool,
            default=True,
            help='EMA feature updating mechanism'
        )
    parser.add_argument(
            '--MC',dest='MC',type=bool,
            default=True,
            help='Matching with both appearance and motion cost'
        )
    parser.add_argument(
            '--woC',dest='woC',type=bool,
            default=True,
            help='Replace the matching cascade with vanilla matching'
        )
    parser.add_argument(
            '--AFLink',dest='AFLink',type=bool,
            default=True,
            help='Appearance-Free Link'
        )
    parser.add_argument(
            '--GSI',dest='GSI',
            action='store_true',
            help='Gaussian-smoothed Interpolation'
        )
    parser.add_argument(
            '--path_AFLink',dest='path_AFLink',type=str,
            default=paths.AFLINK_CKPT
        )
    parser.add_argument(
            '--EMA_alpha',dest='EMA_alpha',type=float,
            default=0.9
        )
    parser.add_argument(
            '--MC_lambda',dest='MC_lambda',type=float,
            default=0.98
        )
    parser.add_argument(
            '--min_confidence',dest='min_confidence',type=float,
            default=0.2
        )
    parser.add_argument(
            '--nms_max_overlap',dest='nms_max_overlap',type=float,
            default=1.0
        )
    parser.add_argument(
            '--min_detection_height',dest='min_detection_height',type=float,
            default=0
        )
    parser.add_argument(
            '--max_cosine_distance',dest='max_cosine_distance',type=float,
            default=0.35
        )
    parser.add_argument(
            '--nn_budget',dest='nn_budget',type=int,
            default=100
        )
    


         
    
    # ----------------------------------------------------------------- METOR
    # Everything below is new in this repository: arguments for the modules
    # METOR adds (paper Sec. 3.3, 3.4) and its training objective (Sec. 3.5).
    # Defaults are the paper's stated values wherever the paper states one;
    # where it does not, the default is marked PAPER-GAP in the help text and
    # justified in docs/30_Paper-gaps.md.
    metor = parser.add_argument_group(
        'METOR', 'Modules and hyperparameters specific to METOR (arXiv 2505.06663)')

    metor.add_argument('--arch', type=str, default='metor',
                       choices=['metor', 'metor-compat'],
                       help="'metor' = the paper as written (100 queries, COCO "
                            "init, 5 losses, untrainable without a 16GB+ GPU). "
                            "'metor-compat' = METOR's modules on the EOV-MMP "
                            "stack, which loads the released checkpoints and "
                            "runs today. See metor/compat.py.")

    # --- Sec. 3.4, iterative enhancement
    metor.add_argument('--n_iter', type=int, default=2,
                       help='N_i, the number of iterative enhancement layers. '
                            'Paper Sec. 4.2: 2 for VidVRD, 3 for VidOR. '
                            '0 reproduces their Table 4 ablation baseline.')
    metor.add_argument('--alpha', type=float, default=0.9,
                       help='alpha of Eq. 11, the object/relationship blend. '
                            'Paper Sec. 4.2: 0.9.')
    metor.add_argument('--residual_mapping', action='store_true',
                       help='DEVIATION from Eq. 11: let M_f predict a delta so '
                            'the layer starts as the identity. See '
                            'metor/modules/iterative_enhancement.py.')
    metor.add_argument('--st_max_len', type=int, default=512,
                       help='Max sequence length for the spatio-temporal '
                            'Transformer positional embeddings.')

    # --- Sec. 3.3, contextual refinement encoding
    metor.add_argument('--cre_mode', type=str, default='inject',
                       choices=['inject', 'pool'],
                       help="'inject' appends the context tokens to the frozen "
                            "CLIP ViT input, which is what Eq. 7 describes. "
                            "'pool' attention-pools cached patch embeddings -- "
                            "much cheaper, an approximation, NOT equivalent.")
    metor.add_argument('--n_ctx_obj', type=int, default=8,
                       help='Number of object context tokens c_o. METOR does '
                            'not state it, but both predecessors do: MMP and '
                            'EOV-MMP each use "eight tokens for conditional '
                            'prompts", which is the role M_o(C_o) plays in '
                            'Eq. 9. Default 8 on that evidence.')
    metor.add_argument('--n_ctx_rel', type=int, default=8,
                       help='Number of relationship context tokens c_r. Same '
                            'evidence as --n_ctx_obj.')
    metor.add_argument('--n_prompt_vec', type=int, default=8,
                       help='Size of the learnable vectors v_o / v_r in '
                            'Eq. 9-10. METOR does not state it; MMP states '
                            '"eight tokens for learnable continuous prompts" '
                            'and EOV-MMP repeats it. Default 8.')
    metor.add_argument('--rel_token_position', type=str, default='end',
                       choices=['end', 'three_quarter'],
                       help="Where [REL] sits in the Eq. 10 prompt. 'end' is "
                            "what Eq. 10 prints AND what METOR's Figure 3 "
                            "draws, so it is the correct default. "
                            "'three_quarter' is what both predecessors do "
                            "(MMP and EOV-MMP put [REL] at 75%% of the token "
                            "length); kept only to test that variant. "
                            "See docs/33_Verification-status.md.")
    metor.add_argument('--no_crq', action='store_true',
                       help="Ablation 'w/o CRQ' (their Table 3): skip the "
                            'contextual refinement of object queries, Eq. 8.')
    metor.add_argument('--no_crt', action='store_true',
                       help="Ablation 'w/o CRT' (their Table 3): skip the "
                            'contextual refinement of text features, Eq. 9-10.')

    # --- Sec. 3.2, framework shape
    metor.add_argument('--n_queries', type=int, default=100,
                       help='N_q, number of object queries. Paper Sec. 4.2: 100. '
                            'NOTE EOV-MMP uses 300, which is why its released '
                            'detector cannot be loaded into --arch metor.')
    metor.add_argument('--d_model', type=int, default=256,
                       help='Decoder / enhancement hidden dim. PAPER-GAP: '
                            "unspecified; 256 is Deformable-DETR's.")
    metor.add_argument('--n_heads', type=int, default=8,
                       help='Attention heads. MMP: "The head number of '
                            'multi-head self-attention of Transformer blocks '
                            'is set to 8."')
    metor.add_argument('--n_dec_layers', type=int, default=6,
                       help='Transformer decoder depth for Eq. 2. EOV-MMP: '
                            '"we use six Transformer layers and 300 object '
                            'queries" -- METOR changes the query count to 100 '
                            'but says nothing about the depth.')
    metor.add_argument('--dropout', type=float, default=0.1,
                       help='MMP: "the dropout rate is set to 0.1."')
    metor.add_argument('--max_pair_objects', type=int, default=10,
                       help='PAPER-GAP: how many top-scoring queries are paired '
                            'up for relationship classification. The paper says '
                            'only that pairs are "derived" from the object '
                            'features; all pairs of 100 queries would be 9,900.')

    # --- Sec. 3.5, training objective
    metor.add_argument('--theta_traj', type=float, default=1.0,
                       help='Weight of L_traj in Eq. 12. Paper Sec. 4.2: 1.0.')
    metor.add_argument('--theta_ctx', type=float, default=0.2,
                       help='Weight of the contextual losses in Eq. 12. '
                            'Paper Sec. 4.2: 0.2.')
    metor.add_argument('--theta_cst', type=float, default=0.1,
                       help='Weight of the trajectory consistency term in '
                            'Eq. 15. Paper Sec. 4.2: 0.1.')

    # --- RB1: clean EOV-MMP baseline (pre-registration A23). Every flag below
    # defaults to the released behaviour.
    rb1 = parser.add_argument_group(
        'RB1', 'Switches for the clean-baseline batch; all off = EOV-MMP as released')
    rb1.add_argument('--exclude_novel_train', action='store_true',
                     help='Step 4 trains without any novel object category or '
                          'novel predicate: detector boxes, relation instances '
                          'that touch a novel object or predicate, and so the '
                          'interactiveness target. models/novel_filter.py.')
    rb1.add_argument('--uniform_fusion', action='store_true',
                     help='Fuse detector and classifier scores with 0.3/0.7 for '
                          'all 35 classes instead of the released per-class '
                          'weights. models/fusion.py.')
    rb1.add_argument('--no_epoch_eval', action='store_true',
                     help='cli/train.py: do not evaluate on test after each '
                          'epoch (no model selection on test). The last epoch '
                          'is the final model.')
    rb1.add_argument('--epoch_ckpt_dir', type=str, default='',
                     help='cli/train.py: write a full-state checkpoint (model, '
                          'optimizer, scheduler, epoch, RNG states) here at the '
                          'end of every epoch. Empty = off.')
    rb1.add_argument('--mirror_dir', type=str, default='',
                     help='cli/train.py: after every epoch also copy the '
                          'checkpoint and logs to this directory (persistent '
                          'storage outside the machine). Empty = off.')
    rb1.add_argument('--loader_seed', type=int, default=-1,
                     help='cli/train.py: seed of a dedicated generator for the '
                          'training DataLoader shuffle. -1 = the global RNG, as '
                          'released.')
    rb1.add_argument('--expect_n_iter', type=int, default=-1,
                     help='cli/train.py: abort at startup unless the relation '
                          "model's iterative enhancement module has exactly this "
                          'many layers (0 = none). -1 = no check. Guards the '
                          'compat default of --n_iter 2.')
    rb1.add_argument('--small_train_videos', type=int, default=0,
                     help='cli/train.py: train on N videos evenly spaced over '
                          'the sorted training list (plumbing runs). 0 = all.')

    args = parser.parse_args()
    return args 