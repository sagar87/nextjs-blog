import Link from "next/link";
import getPostMetaData from "@/components/getPostMetaData";
import PostPreview from "@/components/PostPreview";

const HomePage = () => {
  const postData = getPostMetaData();
  const postPreview = postData.map((post) => (
    <PostPreview key={post.slug} {...post} />
  ));
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">{postPreview}</div>
  );
};

export default HomePage;
