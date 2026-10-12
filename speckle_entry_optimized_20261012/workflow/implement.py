from pathlib import Path
import re

root=Path(__file__).resolve().parents[1]
src=root/'cpu_cpp/source'
def edit(path, f):
    s=path.read_text(encoding='utf-8'); path.write_text(f(s),encoding='utf-8')

# Both original owning constructors remain available. Borrowing is synchronous,
# read-only, and explicitly limited to the established dense no-dark-field path.
for name, marker, sigold, signew in [
 ('wsvt', 'AlignedVector<float> WSVT::stack_TemplateWindow',
  'std::vector<float>&& img_stack,\n    std::vector<float>&& ref_stack,',
  'std::span<const float> img_stack,\n    std::span<const float> ref_stack,'),
 ('wxst', 'PyramidResult WXST::pyramid_data()',
  'const std::vector<float>& img,\n    const std::vector<float>& ref,',
  'std::span<const float> img,\n    std::span<const float> ref,')]:
    cls=name.upper(); cpp=src/f'src/{name}_pipeline.cpp'; hp=src/f'include/wsvt/{name}_pipeline.hpp'
    s=cpp.read_text(encoding='utf-8')
    start=s.index(f'{cls}::{cls}(\n    '+sigold.split(',\n')[0])
    end=s.index(marker,start)
    ctor=s[start:end].replace(sigold,signew,1)
    if name=='wsvt':
        ctor=ctor.replace(': img_data_(std::move(img_stack)),\n      ref_data_(std::move(ref_stack)),', ': borrowed_img_(img_stack),\n      borrowed_ref_(ref_stack),\n      borrowed_(true),')
        ctor=ctor.replace('img_data_.size()', 'borrowed_img_.size()').replace('ref_data_.size()', 'borrowed_ref_.size()')
        guard='crop_ != 0 || n_template_ != 0 || calc_darkfield_ || use_estimate_ || use_gpu_'
    else:
        ctor=ctor.replace(': img_data_(img),\n      ref_data_(ref),', ': borrowed_img_(img),\n      borrowed_ref_(ref),\n      borrowed_(true),')
        ctor=ctor.replace('img_data_.size()', 'borrowed_img_.size()').replace('ref_data_.size()', 'borrowed_ref_.size()')
        guard='(m_image_ > 0 && static_cast<std::size_t>(m_image_) < std::min(h_, w_)) || use_estimate_ || use_gpu_'
    ctor=ctor.replace('    validate_manual_window_contract(', f'    if ({guard}) throw std::invalid_argument("Borrowed {cls} requires uncropped CPU inputs without estimate/dark-field");\n    validate_manual_window_contract(',1)
    s=s[:end]+ctor+s[end:]
    if name=='wsvt':
        s=s.replace('    auto p = wsvt::pyramid_data_consume(', '''    if (borrowed_) return wsvt::pyramid_data_borrow(
        borrowed_ref_, borrowed_img_, ch_, h_, w_, pyramid_level_, n_template_,
        PyramidDownsampleMode::Db3Aa, PyramidNormalizationMode::InitialStack);
    auto p = wsvt::pyramid_data_consume(''',1)
        s=s.replace('    const auto processing_t0 =', '''    if (borrowed_ && (easy_to_hard_.adaptive_frames ||
        easy_to_hard_.confidence_aware_refinement || easy_to_hard_.lazy_temporal_descriptors ||
        easy_to_hard_.prefix_compatible_temporal_wavelet || wavelet_guided_umpa_.enabled ||
        scalar_darkfield_.enabled)) throw std::invalid_argument("Borrowed input does not support experimental raw-input stages");
    const auto processing_t0 =''',1)
        # Only the original first-frame transmission expression is altered.
        s=s.replace('std::max(ref_data_[idx], 1.0e-10f)', 'std::max((borrowed_ ? borrowed_ref_ : std::span<const float>(ref_data_))[idx], 1.0e-10f)')
        s=s.replace('const float ratio = img_data_[idx] / den;', 'const float ratio = (borrowed_ ? borrowed_img_ : std::span<const float>(img_data_))[idx] / den;')
    else:
        s=s.replace('    return wsvt::pyramid_data(ref_data_', '''    if (borrowed_) return wsvt::pyramid_data_borrow(borrowed_ref_, borrowed_img_,
        1, h_, w_, pyramid_level_, n_s_, PyramidDownsampleMode::Db3Aa,
        PyramidNormalizationMode::PerLevelFeature);
    return wsvt::pyramid_data(ref_data_''',1)
        s=s.replace('WXSTOutput WXST::solver() {', '''WXSTOutput WXST::solver() {
    if (borrowed_ && scalar_darkfield_.enabled)
        throw std::invalid_argument("Borrowed WXST does not support scalar dark-field");''',1)
        s=s.replace('static_cast<double>(ref_data_[i])', 'static_cast<double>((borrowed_ ? borrowed_ref_ : std::span<const float>(ref_data_))[i])')
        s=s.replace('static_cast<double>(img_data_[i])', 'static_cast<double>((borrowed_ ? borrowed_img_ : std::span<const float>(img_data_))[i])')
    cpp.write_text(s,encoding='utf-8')
    h=hp.read_text(encoding='utf-8')
    decl_start=h.index('    '+cls+'(\n        '+sigold.split(',\n')[0])
    decl_end=h.index(');',decl_start)+2
    decl=h[decl_start:decl_end]
    decl=decl.replace(sigold.replace('\n    ', '\n        '),signew.replace('\n    ', '\n        '))
    h=h[:decl_end]+'\n\n    /// Borrowed inputs must remain valid and immutable until the synchronous call returns.\n'+decl+h[decl_end:]
    h=h.replace('    std::vector<float> img_data_;','    std::span<const float> borrowed_img_, borrowed_ref_;\n    bool borrowed_ = false;\n    std::vector<float> img_data_;',1)
    hp.write_text(h,encoding='utf-8')

p=src/'src/pyramid.cpp'; s=p.read_text(encoding='utf-8')
# Read-only helpers accept views. All original owning call sites and arithmetic stay intact.
for function in ['downsample2x_mean','downsample2x_db3_aa','downsample_by_mode',
                 'stack_template_window_hwd','stack_and_normalize_template_hwd']:
    pattern=r'('+function+r'\(\s*)const std::vector<float>&'
    s,n=re.subn(pattern,r'\1std::span<const float>',s)
    assert n==1,(function,n)

normalize=s[s.index('void normalize_stack_depth_chw_inplace('):s.index('PyramidResult build_pyramid_single(')]
normalize=normalize.replace('normalize_stack_depth_chw_inplace(\n    std::vector<float>& data,', 'normalize_stack_depth_chw_copy(\n    std::span<const float> data,\n    std::span<float> output,')
normalize=normalize.replace('data.size() != ch * h * w','data.size() != ch * h * w || output.size() != data.size()')
normalize=normalize.replace('data[c * h * w + pixel] = stdv', 'output[c * h * w + pixel] = stdv')
build=s[s.index('PyramidResult build_pyramid_single('):s.index('\n}\n\nPyramidResult pyramid_data_consume(')]
build=build.replace('build_pyramid_single(\n    std::vector<float>& data,', 'build_pyramid_borrow_single(\n    std::span<const float> data,')
build=build.replace('    if (normalization_mode ==', '    AlignedBuffer<float> normalized;\n    std::span<const float> raw_level = data;\n    if (normalization_mode ==',1)
build=build.replace('normalize_stack_depth_chw_inplace(data, ch, h, w);', 'normalized.resize(data.size());\n        normalize_stack_depth_chw_copy(data, {normalized.data(), normalized.size()}, ch, h, w);\n        raw_level = {normalized.data(), normalized.size()};')
build=build.replace('    std::vector<float> raw_level = std::move(data);', '    std::vector<float> owned_level;')
build=build.replace('            raw_level = std::move(next);', '            owned_level = std::move(next);\n            raw_level = owned_level;\n            normalized.clear();')
# Copy identical two-section thread management, merge and timing semantics.
consume=s[s.index('PyramidResult pyramid_data_consume('):s.index('RawPyramidResult normalized_raw_pyramid_data(')]
consume=consume.replace('pyramid_data_consume(', 'pyramid_data_borrow(',1).replace('std::vector<float>& ref_data', 'std::span<const float> ref_data').replace('std::vector<float>& img_data','std::span<const float> img_data').replace('build_pyramid_single(', 'build_pyramid_borrow_single(')
at=s.index('\n}\n\nPyramidResult pyramid_data_consume(')
s=s[:at]+'\n'+normalize+build+s[at:]
at=s.index('RawPyramidResult normalized_raw_pyramid_data(')
s=s[:at]+consume+s[at:]
p.write_text(s,encoding='utf-8')
hp=src/'include/wsvt/pyramid.hpp';h=hp.read_text(encoding='utf-8')
for function in ['stack_template_window_hwd','stack_and_normalize_template_hwd']:
    h=re.sub(r'('+function+r'\(\s*)const std::vector<float>&',r'\1std::span<const float>',h)
decl=h[h.index('[[nodiscard]] PyramidResult pyramid_data_consume('):h.index('/// Build normalized raw CHW')]
decl=decl.replace('pyramid_data_consume','pyramid_data_borrow').replace('std::vector<float>& ref_data','std::span<const float> ref_data').replace('std::vector<float>& img_data','std::span<const float> img_data')
h=h.replace('/// Build normalized raw CHW', '/// Read-only views; initial normalization writes directly to an uninitialized private buffer.\n'+decl+'/// Build normalized raw CHW',1)
hp.write_text(h,encoding='utf-8')

driver=(root/'cpu_cpp/cpu_driver_original.cpp').read_text()
driver=driver.replace('std::vector<float> s(sample,sample+count),r(reference,reference+count);', 'if(!sample || !reference || n==0 || h==0 || w==0) throw std::invalid_argument("Invalid input");\n        std::span<const float> s(sample,count),r(reference,count);')
driver=driver.replace('solver(std::move(s),std::move(r),','solver(s,r,')
(root/'cpu_cpp/cpu_driver.cpp').write_text(driver)

gp=root/'gpu_common'
edit(gp/'src/session.cu', lambda s:s.replace('extern "C" {','extern "C" bool speckle_all_finite(const float*,size_t);\n\nextern "C" {',1).replace('const float* p=static_cast<const float*>(a);for(size_t j=0;j<bytes/4;++j)if(!std::isfinite(p[j]))throw', 'if(!speckle_all_finite(static_cast<const float*>(a),bytes/4))throw'))
edit(gp/'CMakeLists.txt', lambda s:s.replace('src/session.cu)', 'src/session.cu src/finite.cpp)').replace('target_link_libraries(speckle_native PRIVATE', 'target_compile_options(speckle_native PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-fno-finite-math-only>")\ntarget_link_libraries(speckle_native PRIVATE',1))
