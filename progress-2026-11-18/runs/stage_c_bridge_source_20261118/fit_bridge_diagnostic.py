"""Fixed linked-checkpoint bridge diagnosis; no collection or optimization."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from research.p1a.native_topology_condition import load_host,load_readout
from research.p1a.stage_a_fourroom import OpenPhysics
from research.p1a.stage_c_paired import factual_support,menu_forecast,paired_energy,replace_support,two_offsets
from research.p1a.stage_c_paired_fit import dataset

@torch.inference_mode()
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--directory',type=Path,required=True)
 parser.add_argument('--destination',type=Path)
 parser.add_argument('--device',default='cuda')
 args=parser.parse_args()
 root=args.directory.resolve()
 destination=args.destination or root/'fit_bridge_diagnostic.json'
 device=torch.device(args.device);torch.set_num_threads(2)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 host=load_host(device);decode=load_readout(device)
 checkpoint=torch.load(root/'paired_unlabeled.pt',map_location=device,weights_only=True)
 if 'final_two_blocks' in checkpoint:
  host.predictor.transformer.layers[-2:].load_state_dict(checkpoint['final_two_blocks'],strict=True)
 offsets=two_offsets(checkpoint['b'],checkpoint['v'])
 fit=json.loads((root/'paired_unlabeled_fit.json').read_text(encoding='utf-8-sig'))
 alignment=json.loads((root/'fit_alignment.json').read_text(encoding='utf-8-sig'))
 mapping=alignment['physical_to_explanation']
 if mapping not in ([0,1],[1,0]):raise ValueError('Fit-only explanation naming must be resolved before diagnosis')
 audit=json.loads((root/'bridge_data_audit.json').read_text(encoding='utf-8-sig'))
 with np.load(root/'lifecycle.npz',allow_pickle=False) as data:
  z,actions,windows,controls,truth,directed,reads=dataset(data,host,device)
  menu_raw=data['fit_menu_a'].copy()
 physics=OpenPhysics(SimpleNamespace(wall_thickness=10,BORDER_SIZE=14,IMG_SIZE=224,door_sizes=torch.full((4,),16.)))
 rows=[]
 for meta in audit['connectors']:
  g=meta['group'];n=meta['factual_blocks'];physical_order=[meta['physical_row_order'].index(name) for name in ('bottom','top')]
  pred=menu_forecast(host,windows[g:g+1],controls[g:g+1],offsets)[0]
  p,e,pending=factual_support(host,z[g,:,:n+1],actions[g,:,:n],offsets,fit['s2_obs'])
  err=(pred.permute(0,2,1,3,4).unsqueeze(1)-truth[g].unsqueeze(2)).square().sum(-1).mean((-1,-2))/(2*fit['s2_bundle'])
  supports=p[:,reads[g]].transpose(0,1)
  penalty=-supports.clamp_min(torch.finfo(supports.dtype).tiny).log()
  error_phys=err[:,physical_order];penalty_phys=penalty[:,physical_order];cost=error_phys+penalty_phys
  def alternatives(matrix):return torch.stack(((matrix[:,0,0]+matrix[:,1,1])*.5,(matrix[:,0,1]+matrix[:,1,0])*.5),-1)
  error_pair=alternatives(error_phys);penalty_pair=alternatives(penalty_phys);joint_pair=alternatives(cost)
  linked=joint_pair.mean(0).min();split=joint_pair.min(-1).values.mean()
  native=paired_energy(pred.unsqueeze(0),truth[g:g+1],supports.unsqueeze(0),directed[g:g+1],fit['s2_bundle'])[0]
  if not torch.allclose(linked,native,rtol=1e-6,atol=1e-6):raise ValueError('Manual objective decomposition disagrees with native paired_energy')
  arrivals=(meta['first_probe_arrival'],meta['second_probe_arrival'])
  source_predictions=torch.stack([pending[:,t-3] for t in arrivals])
  source_arrivals=torch.stack([z[g,:,t] for t in arrivals])
  source_error=(source_predictions-source_arrivals.unsqueeze(-2)).square().sum(-1)/(2*fit['s2_obs'])
  source_error_phys=source_error[:,physical_order]
  source_pair=alternatives(source_error_phys)
  combined=.5*(cost+source_error_phys)
  combined_pair=alternatives(combined)
  combined_linked=combined_pair.mean(0).min()
  combined_split=combined_pair.min(-1).values.mean()
  native_combined=paired_energy(pred.unsqueeze(0),truth[g:g+1],supports.unsqueeze(0),directed[g:g+1],fit['s2_bundle'],source_cost=source_error.unsqueeze(0))[0]
  if not torch.allclose(combined_linked,native_combined,rtol=1e-6,atol=1e-6):raise ValueError('Combined decomposition disagrees with native source_cost')
  decoded_pred=decode(pred).permute(0,2,1,3,4)
  decoded_true=decode(truth[g])[:,physical_order]
  actual=np.stack([np.stack([physics.rollout(np.asarray(anchor,dtype=np.float32),menu_raw[g,r],lock)[:,4::5] for lock in ('bottom','top')]) for r,anchor in enumerate(meta['read_roots'])])
  actual_tensor=torch.as_tensor(actual,device=device)
  xy_error=(decoded_pred.unsqueeze(1)-actual_tensor.unsqueeze(2)).square().sum(-1).mean((-1,-2))
  xy_pairs=alternatives(xy_error)
  events=[]
  pre_near,pre_far=meta['probe_pre_blocks']
  for label,t in (('near_probe_before',pre_near),('near_probe_first_arrival',pre_near+1),('near_probe_second_arrival',pre_near+2),('near_read',meta['read_blocks'][0]),('far_probe_before',pre_far),('far_probe_first_arrival',pre_far+1),('far_probe_second_arrival',pre_far+2),('far_read',meta['read_blocks'][1])):
   events.append({'event':label,'t':t,'support_bottom_top':p[physical_order,t].tolist(),'evidence_bottom_top':e[physical_order,t-3].tolist() if t>=3 else None,'issued_decoded_minus_plus_by_bottom_top':decode(pending[physical_order,t-3]).tolist() if t>=3 else None,'actual_arrival_decoded_bottom_top':decode(z[g,physical_order,t]).tolist()})
  omitted=p.new_full((2,2),.5)
  for t in range(3,n+1):
   if t not in (pre_far+1,pre_far+2):omitted=replace_support(omitted,e[:,t-3])
  row={'group':g,'direction':meta['direction'],'physical_row_order':meta['physical_row_order'],'read_roots':meta['read_roots'],'read_blocks':meta['read_blocks'],'first_probe_evidence_bottom_top':e[physical_order,pre_near-2].tolist(),'far_probe_evidence_bottom_top':e[physical_order,pre_far-2].tolist(),'energy_error_by_root_physical_explanation':error_phys.tolist(),'energy_penalty_by_root_physical_explanation':penalty_phys.tolist(),'energy_joint_by_root_physical_explanation':cost.tolist(),'mapping_alternative_labels':['bottom_minus_top_plus','bottom_plus_top_minus'],'error_alternatives_near_far':error_pair.tolist(),'support_penalty_alternatives_near_far':penalty_pair.tolist(),'joint_alternatives_near_far':joint_pair.tolist(),'linked_energy':linked.item(),'split_root_min_energy':split.item(),'linkage_gap':(linked-split).item(),'linked_best_mapping_index':joint_pair.mean(0).argmin().item(),'best_predictive_mapping_index_by_root':error_pair.argmin(-1).tolist(),'best_joint_mapping_index_by_root':joint_pair.argmin(-1).tolist(),'physical_xy_squared_error_alternatives_near_far':xy_pairs.tolist(),'best_decoded_geometry_mapping_index_by_root':xy_pairs.argmin(-1).tolist(),'probe_decode_truth_xy_rmse':(decoded_true-actual_tensor).square().mean().sqrt().item(),'events':events,'far_read_omit_only_far_probe_updates_bottom_top':omitted[physical_order].tolist()}
  source_rows=[]
  for endpoint,t in enumerate(arrivals):
   energies=source_error_phys[endpoint]
   before=p[physical_order,t-1]
   after=p[physical_order,t]
   assigned=energies[torch.arange(2,device=device),mapping]
   wrong=energies[torch.arange(2,device=device),[1-k for k in mapping]]
   source_rows.append({'endpoint':endpoint,'arrival':t,'source_energy_bottom_top_by_minus_plus':energies.tolist(),'evidence_bottom_top':e[physical_order,t-3].tolist(),'intrinsic_assigned_margin_bottom_top':(wrong-assigned).tolist(),'support_before_bottom_top':before.tolist(),'support_after_bottom_top':after.tolist(),'postwrite_assigned_support_change_bottom_top':(after-before)[torch.arange(2,device=device),mapping].tolist()})
  row.update({'h1_sources':source_rows,'source_energy_by_root_physical_explanation':source_error_phys.tolist(),'source_error_alternatives_near_far':source_pair.tolist(),'source_best_mapping_index_by_root':source_pair.argmin(-1).tolist(),'combined_energy_by_root_physical_explanation':combined.tolist(),'combined_alternatives_near_far':combined_pair.tolist(),'combined_linked_energy':combined_linked.item(),'combined_split_endpoint_min_energy':combined_split.item(),'combined_linkage_gap':(combined_linked-combined_split).item(),'combined_best_common_mapping_index':combined_pair.mean(0).argmin().item(),'combined_best_mapping_index_by_root':combined_pair.argmin(-1).tolist()})
  rows.append(row)
  print(g,meta['direction'],'H1',row['source_best_mapping_index_by_root'],'menu',row['best_predictive_mapping_index_by_root'],'joint',row['best_joint_mapping_index_by_root'],'combined',row['combined_best_mapping_index_by_root'],'common',row['combined_best_common_mapping_index'],'H1_margins',[v['intrinsic_assigned_margin_bottom_top'] for v in source_rows],'postwrite_changes',[v['postwrite_assigned_support_change_bottom_top'] for v in source_rows],flush=True)
 result={'scope':'Linked fixed checkpoint; eight fitting bridges only; no optimizer, heldout remapping, collection or intervention.','physical_order':['bottom','top'],'explanation_order':['minus','plus'],'normalization':'Latent squared L2 averaged over all eight candidates and twelve steps, divided by 2*s2_bundle; support negative log averaged over the two physical histories; roots averaged before assignment minimum.','geometry_limit':'Official linear XY probe of predicted latents is diagnostic; true XY computed only from existing issued menu controls via unchanged OpenPhysics equations. It is not a candidate ranking result.','far_probe_omission_limit':'Skips only the two far east/west diagnostic arrivals; later actual recovery and neutral arrivals still update support.','s2_bundle':fit['s2_bundle'],'s2_obs':fit['s2_obs'],'rows':rows,'mean_linkage_gap':float(np.mean([r['linkage_gap'] for r in rows])),'mean_linked_energy':float(np.mean([r['linked_energy'] for r in rows]))}
 result.update({'directory':str(root),'fit_bridge_assignment':fit['bridge_assignment'],'fit_only_physical_to_explanation':mapping,'combined_normalization':'One half of (existing menu energy including endpoint support penalty + actual first-H1 source squared error divided by 2*s2_obs), then average endpoints and histories before one common permutation minimum. No postwrite H1 penalty is added.','mean_combined_linked_energy':float(np.mean([r['combined_linked_energy'] for r in rows])),'mean_combined_linkage_gap':float(np.mean([r['combined_linkage_gap'] for r in rows])),'intrinsic_correct_sources_out_of_32':sum(v>0 for r in rows for s in r['h1_sources'] for v in s['intrinsic_assigned_margin_bottom_top']),'postwrite_positive_sources_out_of_32':sum(v>0 for r in rows for s in r['h1_sources'] for v in s['postwrite_assigned_support_change_bottom_top'])})
 destination.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
 print('wrote',destination,flush=True)
if __name__=='__main__':main()
