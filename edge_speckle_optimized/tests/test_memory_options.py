import unittest
from speckle_edge.config import RunConfig
from speckle_edge.cli import parser,resolve_config
G=dict(pixel_m=1.625e-6,distance_m=.4845,energy_ev=12400.)
class MemoryOptions(unittest.TestCase):
    def test_rejects_invalid_workspace_and_boolean_chunk(self):
        for kw in ({'wsvt_workspace':'typo'},{'spatial_chunk':True},{'spatial_chunk':0},{'spatial_chunk':101},{'numpy_output_copy':1}):
            with self.subTest(kw=kw),self.assertRaises(ValueError):RunConfig('wsvt',**G,**kw)
    def test_explicit_old_workspace_is_available_without_operator_reorder(self):
        c=RunConfig('wsvt',**G,wsvt_workspace='materialized',numpy_output_copy=True)
        self.assertEqual(c.resolved_front,'baseline');self.assertFalse(c.allow_experimental)
        self.assertEqual(c.to_dict()['wsvt_workspace'],'materialized')
    def test_cli_workspace_and_copy_overrides(self):
        base=['wsvt','--input','input.npz','--out','out','--pixel-m','0.000001625','--distance-m','0.4845','--energy-ev','12400']
        c=resolve_config(parser().parse_args(base+['--wsvt-workspace','materialized','--numpy-output-copy']))
        self.assertTrue(c.numpy_output_copy);self.assertEqual(c.wsvt_workspace,'materialized')
        c=resolve_config(parser().parse_args(base+['--no-numpy-output-copy']))
        self.assertFalse(c.numpy_output_copy);self.assertEqual(c.spatial_chunk,16)
if __name__=='__main__':unittest.main()
