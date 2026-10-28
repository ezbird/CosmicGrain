#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

// Ephemeral geometry ONLY. The caller supplies the original distance operation
// and current gas eligibility/mass; no mass, weight or thermodynamics is cached.
// Supported geometry: a cubic periodic box spanning the unsigned integer range.
namespace cg_rc1_geometry {
template<class UInt> class Tree {
public:
  struct Entry {std::array<UInt,3> x; std::uint64_t id; int slot;};
private:
  static_assert(std::is_unsigned<UInt>::value,"Unsigned position type required");
  struct Node {std::array<UInt,3> lo,hi; int begin,end,left,right;};
  std::vector<Entry> entries;
  std::vector<Node> nodes;
  int root=-1;
  int construct(int begin,int end) {
    Node n; n.begin=begin;n.end=end;n.left=n.right=-1;
    n.lo=n.hi=entries[begin].x;
    for(int i=begin+1;i<end;++i)for(int k=0;k<3;++k) {
      n.lo[k]=std::min(n.lo[k],entries[i].x[k]);
      n.hi[k]=std::max(n.hi[k],entries[i].x[k]);
    }
    int index=(int)nodes.size();nodes.push_back(n);
    if(end-begin>16) {
      int axis=0;
      for(int k=1;k<3;++k)if(n.hi[k]-n.lo[k]>n.hi[axis]-n.lo[axis])axis=k;
      int middle=begin+(end-begin)/2;
      std::nth_element(entries.begin()+begin,entries.begin()+middle,entries.begin()+end,
        [axis](const Entry&a,const Entry&b){return a.x[axis]<b.x[axis] ||
          (a.x[axis]==b.x[axis] && a.id<b.id);});
      int left=construct(begin,middle),right=construct(middle,end);
      nodes[index].left=left;nodes[index].right=right;
    }
    return index;
  }
  static UInt separation(UInt a,UInt b) {
    return std::min(UInt(a-b),UInt(b-a));
  }
  template<class Metric> double bound(const Node&n,const UInt*q,Metric metric)const {
    UInt closest[3];
    for(int k=0;k<3;++k) {
      if(q[k]>=n.lo[k] && q[k]<=n.hi[k])closest[k]=q[k];
      else closest[k]=separation(q[k],n.lo[k])<=separation(q[k],n.hi[k])?n.lo[k]:n.hi[k];
    }
    return metric(closest,q);
  }
  static bool outside(double lower,double limit) {
    // Conservative slack at floating boundaries: extra leaves are harmless.
    // Exact inclusion and tie decisions use the caller's original metric.
    return lower>limit && lower-limit>1e-12*std::max(1.0,std::max(lower,limit));
  }
  template<class Metric,class Visit> void radius_node(int i,const UInt*q,double r2,Metric metric,Visit visit)const {
    const Node&n=nodes[i];if(outside(bound(n,q,metric),r2))return;
    if(n.left<0) {
      for(int p=n.begin;p<n.end;++p) {
        double d2=metric(entries[p].x.data(),q);
        if(d2<=r2)visit(entries[p],d2);
      }
    } else {radius_node(n.left,q,r2,metric,visit);radius_node(n.right,q,r2,metric,visit);}
  }
  template<class Metric,class Eligible> void nearest_node(int i,const UInt*q,Metric metric,Eligible eligible,
      bool &found,Entry &best,double &r2)const {
    const Node&n=nodes[i];if(found && outside(bound(n,q,metric),r2))return;
    if(n.left<0) {
      for(int p=n.begin;p<n.end;++p) {
        const Entry&e=entries[p];if(!eligible(e))continue;
        double d2=metric(e.x.data(),q);
        if(!found || d2<r2 || (d2==r2 && e.id<best.id)){found=true;best=e;r2=d2;}
      }
    } else {
      int first=n.left,second=n.right;
      if(bound(nodes[second],q,metric)<bound(nodes[first],q,metric))std::swap(first,second);
      nearest_node(first,q,metric,eligible,found,best,r2);
      nearest_node(second,q,metric,eligible,found,best,r2);
    }
  }
public:
  void build(std::vector<Entry> data) {
    if(data.size()>size_t(std::numeric_limits<int>::max()/2))throw std::runtime_error("AGB geometry too large");
    entries.swap(data);nodes.clear();nodes.reserve(entries.size()/4+1);
    root=entries.empty()?-1:construct(0,(int)entries.size());
  }
  template<class Check> void validate(Check check)const {for(const auto&e:entries)check(e);}
  size_t size()const{return entries.size();}
  template<class Metric,class Eligible> bool nearest(const UInt*q,Metric metric,Eligible eligible,Entry&best,double&r2)const {
    bool found=false;r2=std::numeric_limits<double>::max();
    if(root>=0)nearest_node(root,q,metric,eligible,found,best,r2);
    return found;
  }
  template<class Metric,class Visit> void radius(const UInt*q,double r2,Metric metric,Visit visit)const {
    if(root>=0)radius_node(root,q,r2,metric,visit);
  }
};
}
